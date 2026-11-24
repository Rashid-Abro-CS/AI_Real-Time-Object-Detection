"""
Tracks which object CLASSES appear together in the same camera scene over
time (co-occurrence) and stores sessions in MongoDB once a given
combination stops being fully visible.

IMPORTANT: this module reports co-occurrence only -- e.g. "Person and Cell
Phone were visible together." It never infers a semantic interaction such
as "person is holding the phone"; that would require a separate,
specialized model.
"""

import datetime
import uuid
import logging

import config

logger = logging.getLogger("relationship")


class _ActiveGroup:
    def __init__(self, objects, timestamp):
        self.objects = objects  # sorted tuple of class names
        self.session_uid = uuid.uuid4().hex
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.missed_frames = 0

    def update(self, timestamp):
        self.last_seen = timestamp
        self.missed_frames = 0

    def to_document(self):
        duration = (self.last_seen - self.first_seen).total_seconds()
        return {
            "objects": list(self.objects),
            "session_uid": self.session_uid,
            "date": self.first_seen.strftime("%Y-%m-%d"),
            "first_seen": self.first_seen.strftime("%H:%M:%S"),
            "last_seen": self.last_seen.strftime("%H:%M:%S"),
            "duration_seconds": round(duration, 2),
        }


class RelationshipManager:
    def __init__(self, db):
        self.db = db
        self._active = {}  # frozenset(class_names) -> _ActiveGroup

    def update(self, current_objects, timestamp=None):
        timestamp = timestamp or datetime.datetime.now()
        # Body/hand regions are useful detections but not ordinary scene objects;
        # keep them out of co-occurrence groups to avoid noisy Person + Arm groups.
        body_tokens = ("hand", "finger", "arm", "leg", "head", "torso", "wrist",
                       "elbow", "shoulder", "knee", "ankle", "eye", "ear", "nose")
        class_names = sorted({obj["class_name"] for obj in current_objects
                              if not any(token in obj["class_name"].lower() for token in body_tokens)})

        active_key = None
        if len(class_names) >= config.MIN_OBJECTS_FOR_RELATIONSHIP:
            active_key = frozenset(class_names)
            if active_key in self._active:
                self._active[active_key].update(timestamp)
            else:
                self._active[active_key] = _ActiveGroup(tuple(class_names), timestamp)

        finished_keys = []
        for key, group in self._active.items():
            if key != active_key:
                group.missed_frames += 1
                gap = (timestamp - group.last_seen).total_seconds()
                if (gap >= config.RELATIONSHIP_GRACE_SECONDS
                        and group.missed_frames >= config.RELATIONSHIP_MIN_MISSED_FRAMES):
                    finished_keys.append(key)

        for key in finished_keys:
            self._close_group(key)

    def _close_group(self, key):
        group = self._active.pop(key, None)
        if group is None:
            return
        duration = (group.last_seen - group.first_seen).total_seconds()
        if duration < config.MIN_RELATIONSHIP_DURATION_SECONDS:
            return
        doc = group.to_document()
        self.db.save_relationship_session(doc)
        logger.info("Saved relationship session: %s", doc)

    def current_group(self):
        """The combination currently fully visible right now, if any."""
        for group in self._active.values():
            if group.missed_frames == 0:
                return list(group.objects)
        return []

    def close_all(self):
        for key in list(self._active.keys()):
            self._close_group(key)
