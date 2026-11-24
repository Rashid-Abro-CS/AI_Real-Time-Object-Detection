"""
MongoDB access layer. Every read/write in the project goes through this
module so the rest of the app never touches PyMongo directly.

Handles connection errors gracefully: if MongoDB is unreachable, the app
keeps running (the camera loop is never blocked) -- writes are simply
skipped and logged until the connection comes back.
"""

import datetime
import collections
import logging
import time
import re

from pymongo import MongoClient, DESCENDING, ASCENDING
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

import config

logger = logging.getLogger("database")


class Database:
    def __init__(self, uri=config.MONGO_URI, db_name=config.MONGO_DB_NAME):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None
        self.connected = False
        self._next_retry_at = 0.0
        self._retry_delay = 2.0
        self._retry_max_delay = 30.0
        self._pending_writes = collections.deque()
        self._pending_limit = 5000
        self._connect()

    # ------------------------------------------------------------------
    def _connect(self):
        try:
            self.client = MongoClient(
                self.uri, serverSelectionTimeoutMS=1000,
                connectTimeoutMS=1000, socketTimeoutMS=1000,
                maxPoolSize=5,
            )
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self._ensure_indexes()
            self.connected = True
            self._retry_delay = 2.0
            self._next_retry_at = 0.0
            logger.info("Connected to MongoDB successfully (db=%s)", self.db_name)
            self._flush_pending()
            # logger.info("Connected to MongoDB at %s (db=%s)", self.uri, self.db_name)
        except (PyMongoError, ServerSelectionTimeoutError) as exc:
            self.connected = False
            self._next_retry_at = time.monotonic() + self._retry_delay
            self._retry_delay = min(self._retry_delay * 2, self._retry_max_delay)
            if self.client:
                try:
                    self.client.close()
                except Exception:
                    pass
            self.client = None
            self.db = None
            logger.warning(
                "Could not connect to MongoDB (%s). Retry will be delayed; camera processing continues.",
                exc,
            )

    def _ensure_indexes(self):
        self.db[config.COLLECTION_DETECTIONS].create_index(
            [("object_name", ASCENDING), ("date", DESCENDING)]
        )
        self.db[config.COLLECTION_DETECTIONS].create_index([("tracking_id", ASCENDING)])
        self.db[config.COLLECTION_SCENES].create_index([("scene_name", ASCENDING), ("date", DESCENDING)])
        self.db[config.COLLECTION_RELATIONSHIPS].create_index([("date", DESCENDING)])
        # Additive indexes for run-scoped reports / duplicate protection. Old
        # documents without these fields are unaffected (sparse); a failure here
        # must never stop the app from connecting.
        try:
            self.db[config.COLLECTION_DETECTIONS].create_index([("run_id", ASCENDING)], sparse=True)
            self.db[config.COLLECTION_DETECTIONS].create_index(
                [("session_uid", ASCENDING)], unique=True, sparse=True)
            self.db[config.COLLECTION_SCENES].create_index(
                [("session_uid", ASCENDING)], unique=True, sparse=True)
            self.db[config.COLLECTION_RELATIONSHIPS].create_index(
                [("session_uid", ASCENDING)], unique=True, sparse=True)
        except PyMongoError as exc:
            logger.warning("Could not create run_id/session_uid indexes: %s", exc)

    def _reconnect_if_needed(self):
        if self.connected:
            return True
        if time.monotonic() < self._next_retry_at:
            return False
        self._connect()
        return self.connected

    def _mark_offline(self):
        self.connected = False
        self._next_retry_at = time.monotonic() + self._retry_delay
        self._retry_delay = min(self._retry_delay * 2, self._retry_max_delay)
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None
        self.db = None

    def _queue_write(self, collection_name, document):
        if len(self._pending_writes) >= self._pending_limit:
            logger.error("Offline write queue full; incoming record was not buffered")
            return False
        doc = dict(document)
        doc.setdefault("created_at", datetime.datetime.now(datetime.timezone.utc))
        self._pending_writes.append((collection_name, doc))
        logger.warning("Buffered MongoDB write in memory (queue=%d)", len(self._pending_writes))
        return True

    @property
    def pending_count(self):
        return len(self._pending_writes)

    def _flush_pending(self):
        if not self.connected or self.db is None:
            return
        while self._pending_writes:
            collection_name, document = self._pending_writes[0]
            try:
                self.db[collection_name].insert_one(document)
                self._pending_writes.popleft()
            except PyMongoError as exc:
                # Unique session_uid index makes ambiguous retries idempotent.
                if getattr(exc, "code", None) == 11000:
                    self._pending_writes.popleft()
                    continue
                logger.warning("Could not flush offline write queue: %s", exc)
                self._mark_offline()
                break

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def save_detection_session(self, session: dict):
        """session keys: object_name, tracking_id, date, first_seen, last_seen,
        duration_seconds, max_confidence, avg_confidence"""
        if not self._reconnect_if_needed():
            self._queue_write(config.COLLECTION_DETECTIONS, session)
            return None
        try:
            doc = dict(session)
            doc["created_at"] = datetime.datetime.now(datetime.timezone.utc)
            return self.db[config.COLLECTION_DETECTIONS].insert_one(doc).inserted_id
        except PyMongoError as exc:
            logger.error("Failed to save detection session: %s", exc)
            self._queue_write(config.COLLECTION_DETECTIONS, session)
            self._mark_offline()
            return None

    def save_scene_session(self, session: dict):
        if not self._reconnect_if_needed():
            self._queue_write(config.COLLECTION_SCENES, session)
            return None
        try:
            doc = dict(session)
            doc["created_at"] = datetime.datetime.now(datetime.timezone.utc)
            return self.db[config.COLLECTION_SCENES].insert_one(doc).inserted_id
        except PyMongoError as exc:
            logger.error("Failed to save scene session: %s", exc)
            self._queue_write(config.COLLECTION_SCENES, session)
            self._mark_offline()
            return None

    def save_relationship_session(self, session: dict):
        if not self._reconnect_if_needed():
            self._queue_write(config.COLLECTION_RELATIONSHIPS, session)
            return None
        try:
            doc = dict(session)
            doc["created_at"] = datetime.datetime.now(datetime.timezone.utc)
            return self.db[config.COLLECTION_RELATIONSHIPS].insert_one(doc).inserted_id
        except PyMongoError as exc:
            logger.error("Failed to save relationship session: %s", exc)
            self._queue_write(config.COLLECTION_RELATIONSHIPS, session)
            self._mark_offline()
            return None

    def update_live_state(self, state: dict):
        """Upserts a single document that the dashboard polls for 'current' info."""
        if not self._reconnect_if_needed():
            return None
        try:
            doc = dict(state)
            doc["updated_at"] = datetime.datetime.now(datetime.timezone.utc)
            return self.db[config.COLLECTION_LIVE_STATE].update_one(
                {"_id": "live"}, {"$set": doc}, upsert=True
            )
        except PyMongoError as exc:
            logger.error("Failed to update live state: %s", exc)
            self._mark_offline()
            return None

    def _pending_matches(self, collection_name, query, limit=200):
        results = []
        for queued_collection, doc in self._pending_writes:
            if queued_collection != collection_name:
                continue
            matched = True
            for key, expected in query.items():
                value = doc.get(key)
                if isinstance(expected, dict) and "$regex" in expected:
                    if value is None or not re.search(expected["$regex"], str(value), re.IGNORECASE):
                        matched = False
                        break
                elif isinstance(expected, dict) and "$gte" in expected:
                    if value is None or float(value) < float(expected["$gte"]):
                        matched = False
                        break
                elif isinstance(expected, dict) and "$all" in expected:
                    if value is None or not all(item in value for item in expected["$all"]):
                        matched = False
                        break
                elif value != expected:
                    matched = False
                    break
            if matched:
                results.append(doc)
        results.sort(
            key=lambda d: d.get("created_at", datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)),
            reverse=True,
        )
        return results[:limit]

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_live_state(self):
        if not self._reconnect_if_needed():
            return None
        try:
            return self.db[config.COLLECTION_LIVE_STATE].find_one({"_id": "live"})
        except PyMongoError as exc:
            logger.error("Failed to read live state: %s", exc)
            self._mark_offline()
            return None

    def search_detections(self, object_name=None, tracking_id=None, date=None,
                           min_duration=None, min_confidence=None, limit=200, run_id=None):
        query = {}
        if object_name:
            query["object_name"] = {"$regex": re.escape(object_name), "$options": "i"}
        if tracking_id not in (None, ""):
            try:
                query["tracking_id"] = int(tracking_id)
            except (TypeError, ValueError):
                pass
        if date:
            query["date"] = date
        if run_id:
            query["run_id"] = run_id
        if min_duration:
            query["duration_seconds"] = {"$gte": float(min_duration)}
        if min_confidence:
            query["max_confidence"] = {"$gte": float(min_confidence)}
        if not self._reconnect_if_needed():
            return self._pending_matches(config.COLLECTION_DETECTIONS, query, limit=limit)
        try:
            cursor = self.db[config.COLLECTION_DETECTIONS].find(query).sort(
                "created_at", DESCENDING
            ).limit(limit)
            return list(cursor)
        except PyMongoError as exc:
            logger.error("Search failed: %s", exc)
            self._mark_offline()
            return []

    def search_scenes(self, scene_name=None, date=None, limit=200):
        query = {}
        if scene_name:
            query["scene_name"] = {"$regex": re.escape(scene_name), "$options": "i"}
        if date:
            query["date"] = date
        if not self._reconnect_if_needed():
            return self._pending_matches(config.COLLECTION_SCENES, query, limit=limit)
        try:
            return list(
                self.db[config.COLLECTION_SCENES].find(query).sort("created_at", DESCENDING).limit(limit)
            )
        except PyMongoError as exc:
            logger.error("Search failed: %s", exc)
            self._mark_offline()
            return []

    def search_relationships(self, objects=None, date=None, limit=200):
        query = {}
        if objects:
            query["objects"] = {"$all": objects}
        if date:
            query["date"] = date
        if not self._reconnect_if_needed():
            return self._pending_matches(config.COLLECTION_RELATIONSHIPS, query, limit=limit)
        try:
            return list(
                self.db[config.COLLECTION_RELATIONSHIPS].find(query).sort("created_at", DESCENDING).limit(limit)
            )
        except PyMongoError as exc:
            logger.error("Search failed: %s", exc)
            self._mark_offline()
            return []

    def get_statistics(self):
        if not self._reconnect_if_needed():
            detection_docs = [d for c, d in self._pending_writes if c == config.COLLECTION_DETECTIONS]
            scene_docs = [d for c, d in self._pending_writes if c == config.COLLECTION_SCENES]
            if not detection_docs and not scene_docs:
                return {}
            names = [d.get("object_name") for d in detection_docs if d.get("object_name")]
            scene_names = [d.get("scene_name") for d in scene_docs if d.get("scene_name")]
            longest_detection = max(detection_docs, key=lambda d: d.get("duration_seconds", 0), default=None)
            longest_scene = max(scene_docs, key=lambda d: d.get("duration_seconds", 0), default=None)
            most_detected = max(set(names), key=names.count, default=None)
            most_common_scene = max(set(scene_names), key=scene_names.count, default=None)
            return {
                "total_detection_sessions": len(detection_docs),
                "unique_objects": len(set(names)),
                "most_detected_object": most_detected,
                "longest_detection_seconds": longest_detection.get("duration_seconds", 0) if longest_detection else 0,
                "total_scene_sessions": len(scene_docs),
                "most_common_scene": most_common_scene,
                "longest_scene_name": longest_scene.get("scene_name") if longest_scene else None,
                "longest_scene_seconds": longest_scene.get("duration_seconds", 0) if longest_scene else 0,
            }
        try:
            det = self.db[config.COLLECTION_DETECTIONS]
            scenes = self.db[config.COLLECTION_SCENES]

            total_sessions = det.count_documents({})
            unique_objects = len(det.distinct("object_name"))
            longest_doc = next(det.find().sort("duration_seconds", DESCENDING).limit(1), None)

            most_detected_pipeline = [
                {"$group": {"_id": "$object_name", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 1},
            ]
            most_detected = list(det.aggregate(most_detected_pipeline))

            total_scene_sessions = scenes.count_documents({})
            most_common_scene_pipeline = [
                {"$group": {"_id": "$scene_name", "count": {"$sum": 1}}},
                {"$sort": {"count": -1}},
                {"$limit": 1},
            ]
            most_common_scene = list(scenes.aggregate(most_common_scene_pipeline))
            longest_scene = next(scenes.find().sort("duration_seconds", DESCENDING).limit(1), None)

            return {
                "total_detection_sessions": total_sessions,
                "unique_objects": unique_objects,
                "most_detected_object": most_detected[0]["_id"] if most_detected else None,
                "longest_detection_seconds": longest_doc["duration_seconds"] if longest_doc else 0,
                "total_scene_sessions": total_scene_sessions,
                "most_common_scene": most_common_scene[0]["_id"] if most_common_scene else None,
                "longest_scene_name": longest_scene["scene_name"] if longest_scene else None,
                "longest_scene_seconds": longest_scene["duration_seconds"] if longest_scene else 0,
            }
        except PyMongoError as exc:
            logger.error("Failed to compute statistics: %s", exc)
            self._mark_offline()
            return {}

    def get_relationship_totals(self, limit=20):
        """Aggregate total time each object combination has spent together."""
        if not self._reconnect_if_needed():
            grouped = {}
            for collection, doc in self._pending_writes:
                if collection != config.COLLECTION_RELATIONSHIPS:
                    continue
                key = tuple(sorted(doc.get("objects", [])))
                item = grouped.setdefault(key, {"_id": list(key), "total_seconds": 0.0, "sessions": 0})
                item["total_seconds"] += float(doc.get("duration_seconds", 0))
                item["sessions"] += 1
            return sorted(grouped.values(), key=lambda d: d["total_seconds"], reverse=True)[:limit]
        try:
            pipeline = [
                {"$group": {"_id": "$objects", "total_seconds": {"$sum": "$duration_seconds"},
                             "sessions": {"$sum": 1}}},
                {"$sort": {"total_seconds": -1}},
                {"$limit": limit},
            ]
            return list(self.db[config.COLLECTION_RELATIONSHIPS].aggregate(pipeline))
        except PyMongoError as exc:
            logger.error("Failed to aggregate relationships: %s", exc)
            self._mark_offline()
            return []

    def clear_history(self):
        if not self._reconnect_if_needed():
            return False
        try:
            self.db[config.COLLECTION_DETECTIONS].delete_many({})
            self.db[config.COLLECTION_SCENES].delete_many({})
            self.db[config.COLLECTION_RELATIONSHIPS].delete_many({})
            return True
        except PyMongoError as exc:
            logger.error("Failed to clear history: %s", exc)
            self._mark_offline()
            return False

    def close(self):
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
        self.connected = False
