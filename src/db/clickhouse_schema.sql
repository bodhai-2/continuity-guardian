-- Continuity Guardian: ClickHouse schema
-- Designed for fast analytical queries across scenes/takes (e.g. "every take
-- where object X changed position") rather than transactional lookups.

CREATE DATABASE IF NOT EXISTS continuity_guardian;

CREATE TABLE IF NOT EXISTS continuity_guardian.shots
(
    shot_id UUID DEFAULT generateUUIDv4(),
    scene_id String,
    take_number UInt16,
    source_file String,
    shot_summary String,
    thumbnail_b64 String DEFAULT '',
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (scene_id, take_number);

-- Safe to re-run against a database created before thumbnails existed.
ALTER TABLE continuity_guardian.shots ADD COLUMN IF NOT EXISTS thumbnail_b64 String DEFAULT '';

CREATE TABLE IF NOT EXISTS continuity_guardian.detections
(
    detection_id UUID DEFAULT generateUUIDv4(),
    shot_id UUID,
    scene_id String,
    take_number UInt16,
    entity_type Enum8('object' = 1, 'actor' = 2),
    label String,
    screen_position String,
    state String,
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (scene_id, label, take_number);

CREATE TABLE IF NOT EXISTS continuity_guardian.continuity_flags
(
    flag_id UUID DEFAULT generateUUIDv4(),
    scene_id String,
    take_a UInt16,
    take_b UInt16,
    entity_label String,
    flag_text String,
    severity Enum8('low' = 1, 'medium' = 2, 'high' = 3),
    resolved UInt8 DEFAULT 0,
    ingested_at DateTime DEFAULT now()
)
ENGINE = MergeTree
ORDER BY (scene_id, severity, ingested_at);

CREATE TABLE IF NOT EXISTS continuity_guardian.camera_notes
(
    shot_id UUID,
    scene_id String,
    take_number UInt16,
    framing String,
    screen_direction String,
    lighting String
)
ENGINE = MergeTree
ORDER BY (scene_id, take_number);
