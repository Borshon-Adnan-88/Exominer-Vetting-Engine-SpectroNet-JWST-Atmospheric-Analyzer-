"""Versioned field definitions for the Stage 1 Kepler catalogue."""

DATASET_ID = "kepler_q1_q17_dr25_koi"
SOURCE_TABLE = "q1_q17_dr25_koi"
LABEL_POLICY_NAME = "confirmed_vs_false_positive_v1"
SCHEMA_VERSION = 1

SELECTED_FIELDS = (
    "kepid",
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_disp_prov",
    "koi_comment",
    "koi_vet_stat",
    "koi_vet_date",
    "koi_tce_plnt_num",
    "koi_tce_delivname",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_model_snr",
    "koi_max_mult_ev",
    "koi_num_transits",
    "koi_quarters",
    "koi_count",
    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_srad",
    "koi_smass",
    "ra",
    "dec",
    "koi_kepmag",
)

STRING_FIELDS = (
    "kepoi_name",
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_disp_prov",
    "koi_comment",
    "koi_vet_stat",
    "koi_vet_date",
    "koi_tce_delivname",
    "koi_quarters",
)

INTEGER_FIELDS = (
    "kepid",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_tce_plnt_num",
    "koi_num_transits",
    "koi_count",
)

NUMERIC_FIELDS = tuple(
    field for field in SELECTED_FIELDS if field not in STRING_FIELDS and field not in INTEGER_FIELDS
)

FORBIDDEN_FUTURE_MODEL_FEATURES = (
    "kepler_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_disp_prov",
    "koi_comment",
    "label",
    "label_name",
    "label_policy",
    "inclusion_status",
    "exclusion_reason",
    "label_conflict",
    "conflict_reason",
)

CANDIDATE_FEATURES_PENDING_REVIEW = (
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_model_snr",
    "koi_max_mult_ev",
    "koi_num_transits",
    "koi_quarters",
    "koi_count",
    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_srad",
    "koi_smass",
    "ra",
    "dec",
    "koi_kepmag",
)
