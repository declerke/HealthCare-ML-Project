CREATE TABLE IF NOT EXISTS patients (
    id                 SERIAL PRIMARY KEY,
    name               VARCHAR(120),
    age                INTEGER        NOT NULL,
    gender             VARCHAR(10)    NOT NULL,
    blood_type         VARCHAR(5)     NOT NULL,
    medical_condition  VARCHAR(60)    NOT NULL,
    date_of_admission  DATE,
    doctor             VARCHAR(120),
    hospital           VARCHAR(120),
    insurance_provider VARCHAR(60),
    billing_amount     NUMERIC(12, 2),
    room_number        INTEGER,
    admission_type     VARCHAR(20)    NOT NULL,
    discharge_date     DATE,
    medication         VARCHAR(60),
    test_results       VARCHAR(20)    NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    id                 SERIAL PRIMARY KEY,
    age                INTEGER        NOT NULL,
    gender             VARCHAR(10)    NOT NULL,
    blood_type         VARCHAR(5)     NOT NULL,
    medical_condition  VARCHAR(60)    NOT NULL,
    admission_type     VARCHAR(20)    NOT NULL,
    billing_amount     NUMERIC(12, 2) NOT NULL,
    insurance_provider VARCHAR(60)    NOT NULL,
    medication         VARCHAR(60)    NOT NULL,
    predicted_result   VARCHAR(20)    NOT NULL,
    confidence         NUMERIC(5, 4)  NOT NULL,
    model_version      VARCHAR(50),
    created_at         TIMESTAMP      DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_versions (
    id          SERIAL PRIMARY KEY,
    version     VARCHAR(50)   NOT NULL,
    accuracy    NUMERIC(6, 4) NOT NULL,
    f1_score    NUMERIC(6, 4) NOT NULL,
    n_samples   INTEGER       NOT NULL,
    is_active   BOOLEAN       DEFAULT TRUE,
    trained_at  TIMESTAMP     DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_predictions_result     ON predictions (predicted_result);
CREATE INDEX IF NOT EXISTS idx_model_versions_active  ON model_versions (is_active);
