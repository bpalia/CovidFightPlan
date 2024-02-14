import pandas as pd
import numpy as np
from typing import List, Tuple


def load_cases_data(path: str) -> pd.DataFrame:
    """Function to load data of COVID-19 infection cases in South Korea.
    Available columns:
    * `case_id`: the ID of the infection case. (case_id(7) = region_code(5) +
    case_number(2). Check `region_code` in 'Region.csv')
    * `province`: Special City / Metropolitan City / Province(-do)
    * `city`: City(-si) / Country (-gun) / District (-gu) (The value 'from other city'
    means that where the group infection started is other city.)
    * `group`: TRUE: group infection / FALSE: not group (If the value is 'TRUE'
    in this column, the value of 'infection_cases' means the name of group. The values
    named 'contact with patient', 'overseas inflow' and 'etc' are not group infection.)
    * `infection_case`: the infection case (the name of group or other cases). (The value
    'overseas inflow' means that the infection is from other country. The value 'etc'
    includes individual cases, cases where relevance classification is ongoing
    after investigation, and cases under investigation.)
    * `confirmed`: the accumulated number of the confirmed
    * `latitude`: the latitude of the group (WGS84)
    * `longitude`: the longitude of the group (WGS84)
    """
    df = pd.read_csv(path)
    print(path + " successfully loaded into dataframe.")
    df = df.rename(columns=lambda x: x.strip())

    # fill in missing coordinates and create substitute city (province name)
    # if actual city is not provided or infection is from other city
    df.loc[df["latitude"] == "-", ["latitude", "longitude"]] = np.nan
    without_city = df["city"] == "-"  # mask
    df.loc[without_city, "city"] = df.loc[without_city, "province"]
    df["sub_city"] = df["city"]
    other_city = df["city"] == "from other city"  # mask
    df.loc[other_city, "sub_city"] = df.loc[other_city, "province"]
    print("Helper column 'sub_city' created.")

    # change data types of other columns
    df = df.astype({"case_id": str, "latitude": float, "longitude": float})

    print("\nData types of each column:")
    print(df.dtypes)
    return df


def load_patient_info(path: str) -> pd.DataFrame:
    """Function to load epidemiological data of COVID-19 patients
    in South Korea. Available columns:
    * `patient_id`: the ID of the patient (patient_id(10) =
    region_code(5) + patient_number(5). Check `region_code` in 'Region.csv')
    * `sex`: the sex of the patient (male / female)
    * `age`: the age of the patient in decades
    * `country`: the country of the patient
    * `province`: the province of the patient (Special City / Metropolitan City /
    Province(-do))
    * `city`: the city of the patient (City(-si) / Country (-gun) / District (-gu))
    * `infection_case`: the case of infection
    * `infected_by`: the ID of who infected the patient (This
    column refers to the `patient_id` column.)
    * `contact_number`: the number of contacts with people
    * `symptom_onset_date`: the date of symptom onset
    * `confirmed_date`: the date of being confirmed
    * `released_date`: the date of being released
    * `deceased_date`: the date of being deceased
    * `state`: isolated / released / deceased (isolation and release in and from Hospital)
    """

    # load data while parsing dates
    df = pd.read_csv(
        path,
        parse_dates=[
            "symptom_onset_date",
            "confirmed_date",
            "released_date",
            "deceased_date",
        ],
    )
    df["symptom_onset_date"] = (
        df["symptom_onset_date"].str.strip().astype("datetime64")
    )
    print(path + " successfully loaded into dataframe.")

    # deal with contact_number if - or patient ID provided
    df.loc[df["contact_number"] == "-", "contact_number"] = np.nan
    df.loc[df["contact_number"].str.len() == 10, "contact_number"] = np.nan

    # change data types of other columns
    age_dtype = pd.CategoricalDtype(
        categories=[
            "0s",
            "10s",
            "20s",
            "30s",
            "40s",
            "50s",
            "60s",
            "70s",
            "80s",
            "90s",
            "100s",
        ],
        ordered=True,
    )
    df = df.astype(
        {
            "patient_id": str,
            "age": age_dtype,
            "sex": "category",
            "state": "category",
            "contact_number": "float32",
        }
    )

    # create age category column
    age_cat_dtype = pd.CategoricalDtype(
        categories=["young", "middle", "old"], ordered=True
    )
    df["age_category"] = (
        df["age"].apply(lambda x: assign_age_category(x)).astype(age_cat_dtype)
    )
    print(
        "Column 'age_category' created. Defines young (0-29), middle (30-59), and old (60+)."
    )

    # calculate intervals between relevant dates
    df["symptom_to_confirmed"] = df.apply(
        lambda x: (x.confirmed_date - x.symptom_onset_date).days, axis=1
    ).astype("float32")
    print(
        "Column 'symptom_to_confirmed' created. Defines number of days from symptom onset to confirmation."
    )
    df["confirmed_to_released"] = df.apply(
        lambda x: (x.released_date - x.confirmed_date).days, axis=1
    ).astype("float32")
    print(
        "Column 'confirmed_to_released' created. Defines number of days from confirmation to release."
    )
    df["confirmed_to_deceased"] = df.apply(
        lambda x: (x.deceased_date - x.confirmed_date).days, axis=1
    ).astype("float32")
    print(
        "Column 'confirmed_to_deceased' created. Defines number of days from confirmation to decease."
    )

    # set index to patient ID
    df.set_index("patient_id", inplace=True)
    print("Dataframe index set to 'patient_id'.")
    print("\nData types of each column:")
    print(df.dtypes)

    return df


def update_time_df(
    df: pd.DataFrame,
    expand: List[str],
    ratio: Tuple[str, str],
    grouping: str = None,
) -> pd.DataFrame:
    """Function to expand accumulated timeseries data into daily information,
    calculate ratio of interest.
    """
    ratio_column_name = ratio[0] + "_to_" + ratio[1] + "_ratio"

    # build dataframe for daily updates
    if grouping is not None:
        df_new = df.groupby(grouping)[expand].diff()
    else:
        df_new = df[expand].diff()

    df_new = df_new.fillna(value=df)
    df_new = df_new.astype(int)
    df_new[ratio_column_name] = df_new[ratio[0]] / df_new[ratio[1]]
    df_new = df_new.add_prefix("new_")

    # update old dataframe
    df[ratio_column_name] = df[ratio[0]] / df[ratio[1]]
    expand.append(ratio_column_name)
    for column in expand:
        df.rename(columns={column: "accumulated_" + column}, inplace=True)

    # concat both
    df = pd.concat([df, df_new], axis=1)

    return df


def assign_age_category(x):
    """Small function to infer age category.
    young: 0-29,
    middle: 30-59,
    old: 60+
    """
    if pd.isnull(x):
        return np.nan
    elif x <= "20s":
        return "young"
    elif x <= "50s":
        return "middle"
    else:
        return "old"
