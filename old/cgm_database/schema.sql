-- This removes everything. You better be careful.
DROP SCHEMA public CASCADE;
CREATE SCHEMA public;

-- Creates a table for person
CREATE TABLE IF NOT EXISTS person (
    id VARCHAR(255) PRIMARY KEY,
    name TEXT NOT NULL,
    surname TEXT NOT NULL,
    birthday BIGINT NOT NULL,
    sex TEXT NOT NULL,
    guardian TEXT NOT NULL,
    is_age_estimated BOOLEAN NOT NULL,
    qr_code TEXT NOT NULL,
    created BIGINT NOT NULL,
    timestamp BIGINT NOT NULL,
    created_by TEXT NOT NULL,
    deleted BOOLEAN NOT NULL,
    deleted_by TEXT NOT NULL
);

-- Creates a table for measure
-- TODO needs location data
CREATE TABLE IF NOT EXISTS measure (
    id VARCHAR(255) PRIMARY KEY,
    person_id TEXT REFERENCES person(id),
    date BIGINT NOT NULL,
    type TEXT NOT NULL,
    age BIGINT NOT NULL,
    height REAL NOT NULL,
    weight REAL NOT NULL,
    muac REAL NOT NULL,
    head_circumference REAL NOT NULL,
    artifact TEXT NOT NULL,
    visible BOOLEAN NOT NULL,
    oedema BOOLEAN NOT NULL,
    timestamp BIGINT NOT NULL,
    created_by TEXT NOT NULL,
    deleted BOOLEAN NOT NULL,
    deleted_by TEXT NOT NULL    
);

-- Creates a table for artifact
CREATE TABLE IF NOT EXISTS artifact (
    id VARCHAR(255) PRIMARY KEY,
    measure_id TEXT REFERENCES measure(id),
    type TEXT NOT NULL,
    path TEXT NOT NULL,
    hash_value TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    upload_date BIGINT NOT NULL,
    deleted BOOLEAN NOT NULL,
    qr_code TEXT NOT NULL,
    create_date BIGINT NOT NULL,
    created_by TEXT NOT NULL,
    status integer NOT NULL
);

-- Creates a table for storing artifact quality assessments.
CREATE TABLE IF NOT EXISTS artifact_quality (
    PRIMARY KEY(artifact_id, type, key),
    type TEXT NOT NULL,
    key TEXT NOT NULL,
    value REAL NOT NULL,
    confidence_value REAL,
    misc TEXT,
    artifact_id VARCHAR(255) REFERENCES artifact(id)
);

-- Creates a table for storing measure quality assessments.
CREATE TABLE IF NOT EXISTS measure_quality (
    PRIMARY KEY(measure_id, type, key),
    type TEXT NOT NULL,
    key TEXT NOT NULL,
    real_value REAL,
    bool_value BOOLEAN,
    text_value TEXT,
    created_by TEXT NOT NULL,
    measure_id VARCHAR(255) REFERENCES measure(id)
);

-- Creates a table for storing artifact results.
CREATE TABLE IF NOT EXISTS artifact_result (
    PRIMARY KEY(artifact_id, model_name, target_key),
    model_name TEXT NOT NULL,
    target_key TEXT NOT NULL,
    value REAL NOT NULL,
    artifact_id VARCHAR(255) REFERENCES artifact(id)
);


-- Creates a view for image data.
DROP VIEW IF EXISTS image_data;
CREATE VIEW image_data AS 
SELECT 
    id AS id
    , path AS path
    , qr_code AS qrcode
    , 0 AS last_updated
    , false AS rejected_by_expert
    , false AS had_error
    , null AS error_message
    --, width_px INTEGER NOT NULL,
    --, height_px INTEGER NOT NULL,
    , aq1.value AS blur_variance
    --, has_face BOOLEAN NOT NULL,
    --, measurement_id INTEGER REFERENCES measurements(id)
    
    FROM artifact
    INNER JOIN artifact_quality aq1 ON aq1.artifact_id=id

    WHERE artifact.type='jpg'
    AND aq1.key='bluriness'
    ;

-- Creates a view for pointcloud data.
DROP VIEW IF EXISTS pointcloud_data;
CREATE VIEW pointcloud_data AS
SELECT 
    id AS id
    , path AS path
    , qr_code AS qrcode
    , 0 AS last_updated
    , false AS rejected_by_expert
    , false AS had_error
    , null AS error_message
    , aq1.value AS number_of_points
    , aq2.value AS confidence_min
    , aq3.value AS confidence_avg
    , aq4.value AS confidence_std
    , aq5.value AS confidence_max
    , aq6.value AS centroid_x
    , aq7.value AS centroid_y
    , aq8.value AS centroid_z
    , aq9.value AS stdev_x
    , aq10.value AS stdev_y
    , aq11.value AS stdev_z
    --, measurement_id INTEGER REFERENCES measurements(id)
    
    FROM artifact
    INNER JOIN artifact_quality aq1 ON aq1.artifact_id=id
    INNER JOIN artifact_quality aq2 ON aq2.artifact_id=id
    INNER JOIN artifact_quality aq3 ON aq3.artifact_id=id
    INNER JOIN artifact_quality aq4 ON aq4.artifact_id=id
    INNER JOIN artifact_quality aq5 ON aq5.artifact_id=id
    INNER JOIN artifact_quality aq6 ON aq6.artifact_id=id
    INNER JOIN artifact_quality aq7 ON aq7.artifact_id=id
    INNER JOIN artifact_quality aq8 ON aq8.artifact_id=id
    INNER JOIN artifact_quality aq9 ON aq9.artifact_id=id
    INNER JOIN artifact_quality aq10 ON aq10.artifact_id=id
    INNER JOIN artifact_quality aq11 ON aq11.artifact_id=id

    WHERE artifact.type='pcd'
    AND aq1.key='number_of_points'
    AND aq2.key='confidence_min'
    AND aq3.key='confidence_avg'
    AND aq4.key='confidence_std'
    AND aq5.key='confidence_max'
    AND aq6.key='centroid_x'
    AND aq7.key='centroid_y'
    AND aq8.key='centroid_z'
    AND aq9.key='stdev_x'
    AND aq10.key='stdev_y'
    AND aq11.key='stdev_z'
    ;

-- Creates a view for ML training.
DROP VIEW IF EXISTS artifacts_with_targets;
CREATE VIEW artifacts_with_targets AS 
SELECT
    person.qr_code AS qr_code,
    artifact.id AS artifact_id,
    artifact.path AS artifact_path,
    artifact.type AS type,
    measure.age AS age,
    measure.height AS height,
    measure.weight AS weight,
    measure.muac AS muac,
    measure.head_circumference AS head_circumference,
    measure_quality.text_value AS status
    FROM artifact
    INNER JOIN measure ON artifact.measure_id=measure.id
    INNER JOIN person ON measure.person_id=person.id
    INNER JOIN measure_quality ON measure_quality.measure_id=measure.id
    WHERE measure.height >= 60
    AND measure.height <= 120
    AND measure.weight >= 2
    AND measure.weight <= 20
    AND measure_quality.key='expert_status'
    ;