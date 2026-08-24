import json

from kepler_catalog.fields import SELECTED_FIELDS


def test_every_manifest_field_has_one_feature_classification():
    registry = json.loads(open("configs/kepler_dr25_field_registry.json", encoding="utf-8").read())
    categories = [
        "identifier", "group_only", "tce_federation", "provenance",
        "candidate_feature_pending_review", "forbidden_future_model_feature",
    ]
    memberships = {field: [] for field in ["source_row_index", *SELECTED_FIELDS]}
    for category in categories:
        for field in registry[category]:
            memberships.setdefault(field, []).append(category)
    assert all(len(categories_for_field) == 1 for categories_for_field in memberships.values())
    assert not (
        set(registry["candidate_feature_pending_review"])
        & set(registry["forbidden_future_model_feature"])
    )


def test_all_label_and_robovetter_audit_fields_are_forbidden():
    registry = json.loads(open("configs/kepler_dr25_field_registry.json", encoding="utf-8").read())
    forbidden = set(registry["forbidden_future_model_feature"])
    required = {
        "kepler_name", "koi_disposition", "koi_pdisposition", "koi_score",
        "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
        "koi_disp_prov", "koi_comment", "label", "label_name", "label_policy",
        "inclusion_status", "exclusion_reason", "label_conflict", "conflict_reason",
    }
    assert required <= forbidden
