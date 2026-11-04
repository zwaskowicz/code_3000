import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    quasi_identifiers = [
        col for col in anon_df.columns
        if col in aux_df.columns
        and col not in ("anon_id", "matched_name", "name")
    ]

    anon_counts = anon_df.groupby(
        quasi_identifiers, dropna=False
    ).size().reset_index(name="anon_count")

    aux_counts = aux_df.groupby(
        quasi_identifiers, dropna=False
    ).size().reset_index(name="aux_count")

    candidates = anon_df.merge(
        aux_df,
        on=quasi_identifiers,
        how="inner",
        suffixes=("_anon", "_aux")
    )

    candidates = candidates.merge(
        anon_counts,
        on=quasi_identifiers,
        how="left"
    ).merge(
        aux_counts,
        on=quasi_identifiers,
        how="left"
    )

    unique_matches = candidates[
        (candidates["anon_count"] == 1)
        & (candidates["aux_count"] == 1)
    ].copy()

    anon_id_col = "anon_id"
    name_col = (
        "matched_name" if "matched_name" in aux_df.columns
        else "name"
    )

    return unique_matches[[anon_id_col, name_col]].rename(
        columns={name_col: "matched_name"}
    ).reset_index(drop=True)



def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    return matches_df["anon_id"].nunique() / len(anon_df)