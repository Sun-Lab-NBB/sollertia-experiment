"""Provides the assets for interfacing with Google Sheet files."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from datetime import datetime

from ataraxis_base_utilities import console
from sollertia_shared_assets import (
    DrugData,
    ImplantData,
    SubjectData,
    SurgeryData,
    InjectionData,
    ProcedureData,
    parse_session_timestamp,
)
from googleapiclient.discovery import Resource, build
from google.oauth2.service_account import Credentials

if TYPE_CHECKING:
    from pathlib import Path

_SUPPORTED_DATE_FORMATS: set[str] = {"%m-%d-%y", "%m-%d-%Y", "%m/%d/%y", "%m/%d/%Y"}
"""Stores the date format schemas supported when parsing dates read from Google Sheet files."""

_DIGIT_PATTERN: re.Pattern[str] = re.compile(r"\d+")
"""A compiled regular expression that matches one or more consecutive digits."""

_GOOGLE_API_MAX_RETRIES: int = 5
"""The number of automatic retries for Google API requests that fail with transient (5xx or 429) errors."""

_REQUIRED_SURGERY_HEADERS: set[str] = {
    # Subject Data headers
    "id",
    "ear punch",
    "sex",
    "genotype",
    "dob",
    "weight (g)",
    "cage #",
    "location housed",
    "status",
    # Procedure Data headers
    "date",
    "start",
    "end",
    "surgeon",
    "protocol",
    "surgery notes",
    "post-op notes",
    "surgery quality",
    # Drug Data headers
    "lrs (ml)",
    "ketoprofen (ml)",
    "buprenorphine (ml)",
    "dexamethasone (ml)",
}
"""Defines all headers (columns) that must exist in a validly formatted surgery log Google Sheet."""

_SURGERY_LOG_DRUGS: dict[str, str] = {
    "Lactated Ringer's Solution": "lrs",
    "Ketoprofen": "ketoprofen",
    "Buprenorphine": "buprenorphine",
    "Dexamethasone": "dexamethasone",
}
"""Maps each drug tracked by the surgery log to the column stem that stores its data. Each drug's administered volume
is stored in the '<stem> (ml)' column and its reference code in the '<stem> code' column."""

_REQUIRED_WATER_RESTRICTION_HEADERS: set[str] = {
    "date",
    "weight (g)",
    "given by:",
    "water given (ml)",
    "behavior",
    "time",
}
"""Defines all headers (columns) that must exist in a validly formatted water restriction log Google Sheet."""


def _convert_date_time_to_timestamp(date: str, time: str) -> int:
    """Converts the input date and time strings to the microseconds-since-UTC-epoch timestamp format used by the
    Sollertia platform.

    Notes:
        The input date and time are human-entered values expressed in the host machine's local time. They are
        interpreted as local time and converted to a UTC timestamp for internal storage.

    Args:
        date: The date string in one of the supported formats: "%m-%d-%y", "%m-%d-%Y", "%m/%d/%y", or "%m/%d/%Y".
        time: The time string in the format "%H:%M".

    Returns:
        The number of microseconds elapsed since UTC epoch onset.

    Raises:
        ValueError: If date or time inputs are an empty string. If the date or time format does not match any of the
            supported input formats.
    """
    if not isinstance(date, str) or not date:
        message = (
            f"Unable to convert the input date and time into the number of microseconds elapsed since UTC epoch onset "
            f"timestamp when parsing Google Sheet data. Expected a non-empty string input for 'date' argument, but "
            f"encountered {date} of type {type(date).__name__}."
        )
        console.error(message=message, error=ValueError)

    if not isinstance(time, str) or not time:
        message = (
            f"Unable to convert the input date and time into the number of microseconds elapsed since UTC epoch onset "
            f"timestamp when parsing Google Sheet data. Expected a non-empty string input for 'time' argument, but "
            f"encountered {time} of type {type(time).__name__}."
        )
        console.error(message=message, error=ValueError)

    try:
        parsed_time = datetime.strptime(time, "%H:%M").time()  # noqa: DTZ007
    except ValueError:
        message = (
            f"Invalid time format encountered when parsing Google Sheet data. Expected the supported time format "
            f"(%H:%M), but encountered {time}."
        )
        console.error(message=message, error=ValueError)

    for date_format in _SUPPORTED_DATE_FORMATS:
        try:
            parsed_date = datetime.strptime(date, date_format).date()  # noqa: DTZ007
            break
        except ValueError:
            continue
    else:
        message = (
            f"Invalid date format encountered when parsing Google Sheet data. Expected one of the supported formats "
            f"({sorted(_SUPPORTED_DATE_FORMATS)}), but encountered {date}."
        )
        console.error(message=message, error=ValueError)

    # Human-entered date and time values use the host machine's local time. Interprets the combined value as local
    # time, then converts it to a UTC microsecond timestamp for internal storage.
    local_datetime = datetime.combine(date=parsed_date, time=parsed_time).astimezone()
    return int(local_datetime.timestamp() * 1_000_000)


def _extract_coordinate_value(substring: str) -> float:
    """Extracts the numeric value from the input stereotactic coordinate substring.

    Args:
        substring: The stereotactic coordinate substring that contains the coordinate value to be extracted.

    Returns:
        The extracted coordinate value.

    Raises:
        ValueError: If the input substring does not contain an extractable anatomical coordinate value.
    """
    # Finds the coordinate number that precedes the anatomical axis designator (AP, ML, DV) and extracts it as a float.
    match = re.search(r"([-+]?\d*\.?\d+)\s*(AP|ML|DV)", substring)

    if match is not None:
        return float(match.group(1))

    message = f"Unable to extract the anatomical coordinate value from the input substring {substring}."
    console.error(message=message, error=ValueError)
    # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
    raise ValueError(message)  # pragma: no cover


def _parse_stereotactic_coordinates(coordinate_string: str) -> tuple[float, float, float]:
    """Parses the AP, ML, and DV stereotactic coordinates from the input coordinate string.

    Notes:
        This function expects the coordinates to be stored as a string formatted as: "-1.8 AP, 2 ML, .25 DV".

    Args:
        coordinate_string: The input string containing the stereotactic coordinates.

    Returns:
        The tuple of 3 floats, each storing the value of the extracted coordinates in the following order: AP, ML, DV.
    """
    ap_coordinate = 0.0
    ml_coordinate = 0.0
    dv_coordinate = 0.0
    for substring in (s.strip() for s in coordinate_string.split(",")):
        if "AP" in substring.upper():
            ap_coordinate = _extract_coordinate_value(substring)
        elif "ML" in substring.upper():
            ml_coordinate = _extract_coordinate_value(substring)
        elif "DV" in substring.upper():
            dv_coordinate = _extract_coordinate_value(substring)

    return ap_coordinate, ml_coordinate, dv_coordinate


def _convert_index_to_column_letter(index: int) -> str:
    """Converts a 0-based column index to an Excel-style (Google Sheet) column letter (A, B, C, ... Z, AA, AB, ...).

    Args:
        index: The 0-based column index to be converted.

    Returns:
        The Excel-style column letter corresponding to the input index.
    """
    result = ""
    while index >= 0:
        remainder = index % 26
        result = chr(65 + remainder) + result  # 65 is ASCII for 'A'
        index = index // 26 - 1
    return result


def _replace_empty_values(row_data: list[str]) -> list[str | None]:
    """Replaces empty cells and cells containing 'n/a', '--', or '---' inside the input row_data list with None.

    Args:
        row_data: The list of cell values read from a Google Sheet row.

    Returns:
        The filtered version of the input list with all empty and placeholder values replaced with None.
    """
    empty_values = {"", "n/a", "--", "---"}
    return [None if cell.strip().lower() in empty_values else cell for cell in row_data]


class SurgeryLog:
    """Interfaces with the Sollertia platform's surgery log Google Sheet.

    Notes:
        This class is purpose-built to work with the specific surgery log format used by the Sollertia platform.

    Args:
        project_name: The name of the project for which to access the surgery log data.
        animal_id: The unique identifier of the animal for which to access the surgery log data.
        credentials_path: The path to the .JSON file containing the service account credentials for accessing the
            surgery log.
        sheet_id: The unique identifier of the Google Sheet that contains the surgery log data.

    Attributes:
        _project_name: The name of the target project.
        _animal_id: The unique identifier of the target animal.
        _sheet_id: The unique identifier of the surgery log Google Sheet.
        _service: The API service instance that bidirectionally interfaces with the surgery log.
        _headers: Maps the surgery log headers (column names) to the Excel-style Google Sheet column letters
            (A, B, etc.).
        _animals: Stores the unique identifiers of all animals whose data is stored in the surgery log.

    Raises:
        ValueError: If the target Google Sheet is not a valid Sollertia platform surgery log.
    """

    def __init__(
        self,
        project_name: str,
        animal_id: int,
        credentials_path: Path,
        sheet_id: str,
    ) -> None:
        # Saves ID data to be reused during later instance method calls.
        self._project_name: str = project_name
        self._animal_id: int = animal_id
        self._sheet_id: str = sheet_id

        # Generates the credentials' object to access the target Google Sheet.
        credentials = Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            filename=str(credentials_path), scopes=("https://www.googleapis.com/auth/spreadsheets",)
        )

        # Uses the credentials' object to build the access service for the target Google Sheet. This service is then
        # used to fetch the sheet data via HTTP request(s). Disables the discovery-document cache, which is unsupported
        # with the installed oauth2client version and only emits a spurious warning when left enabled.
        self._service: Resource = build(
            serviceName="sheets", version="v4", credentials=credentials, cache_discovery=False
        )

        # Retrieves all values stored in the first row of the target sheet tab. Each tab represents a particular
        # project. The first row contains the headers for all data columns stored in the sheet.
        # Extracts the entire first row.
        headers = (
            self._service.spreadsheets()
            .values()
            .get(spreadsheetId=sheet_id, range=f"'{self._project_name}'!1:1")
            .execute(num_retries=_GOOGLE_API_MAX_RETRIES)
        )

        header_values = headers.get("values", [[]])[0]
        if not header_values:
            message = (
                f"Unable to parse the surgery data for the project {project_name} and animal {animal_id}. The first "
                f"row of the target project's surgery log tab appears to be empty instead of containing the "
                f"human-friendly column headers."
            )
            console.error(message=message, error=ValueError)

        self._headers: dict[str, str] = {}
        for index, header in enumerate(header_values):
            column_letter = _convert_index_to_column_letter(index)
            self._headers[str(header).strip().lower()] = column_letter

        missing_headers = [
            required_header
            for required_header in _REQUIRED_SURGERY_HEADERS
            if required_header.lower() not in self._headers
        ]

        if missing_headers:
            missing_headers_str = ", ".join(f"'{header}'" for header in sorted(missing_headers))
            message = (
                f"Unable to parse the surgery data for the project {project_name} and animal {animal_id}. "
                f"The following required headers are missing from the target surgery log file: {missing_headers_str}."
            )
            console.error(message=message, error=ValueError)

        # Retrieves all animal names (IDs) from the 'ID' column. Each ID is z-filled to a five-digit string for
        # sorting to behave predictably. This data is stored as a tuple of IDs.
        id_column = self._get_column_id("id")
        animal_ids = (
            self._service.spreadsheets()
            .values()
            .get(
                spreadsheetId=sheet_id,
                # Reads row 2 onward in column-major order, since row 1 stores the headers.
                range=f"{self._project_name}!{id_column}2:{id_column}",
                majorDimension="COLUMNS",
            )
            .execute(num_retries=_GOOGLE_API_MAX_RETRIES)
        )
        id_list = animal_ids.get("values", [[]])[0]
        self._animals: tuple[str, ...] = tuple(str(animal_id).zfill(5) for animal_id in id_list)
        if not self._animals:
            message = (
                f"Unable to parse the surgery data for the project {project_name} and animal {animal_id}. The ID "
                f"column of the sheet does not contain any animal identifiers, indicating that the surgery log file is "
                f"empty."
            )
            console.error(message=message, error=ValueError)

        # Converts input animal ID to the same format as IDs stored in the id_list generated above for comparison
        formatted_id = str(self._animal_id).zfill(5)

        # Checks if the animal ID exists in the tuple of all known animal IDs. If not, it raises an error.
        if formatted_id not in self._animals:
            message = (
                f"Unable to parse the surgery data for the project {project_name} and animal {animal_id}. The "
                f"target animal's unique identifier is not stored under the 'ID' column of the processed surgery log "
                f"file."
            )
            console.error(message=message, error=ValueError)

    def __del__(self) -> None:
        """Terminates the HTTP connection to the processed surgery log when the instance is garbage-collected."""
        self._service.close()

    def extract_animal_data(self) -> SurgeryData:
        """Extracts and returns the processed animal's surgical intervention data as a SurgeryData object.

        Returns:
            A SurgeryData instance that stores the extracted data.

        Raises:
            ValueError: If the animal's date, time, weight, or cage cells contain malformed or empty values. Date and
                time cells are parsed via _convert_date_time_to_timestamp, while weight and cage cells are parsed via
                float() and int() respectively.
        """
        # Finds the index of the target animal in the ID value tuple to determine the row number to parse from the
        # sheet. The index is modified by 2 because: +1 for 0-indexing to 1-indexing conversion, +1 to account for the
        # header row
        animal_index = self._animals.index(str(self._animal_id).zfill(5))
        row_number = animal_index + 2

        # Retrieves the entire row of data for the target animal
        row_data = (
            self._service.spreadsheets()  # type: ignore[attr-defined]
            .values()
            .get(spreadsheetId=self._sheet_id, range=f"'{self._project_name}'!{row_number}:{row_number}")
            .execute(num_retries=_GOOGLE_API_MAX_RETRIES)
        )

        # Converts the data from dictionary format into a list of strings.
        row_values = row_data.get("values")[0]

        # Replaces empty cells and value placeholders ('n/a', '--', or '---') with None.
        row_values = _replace_empty_values(row_values)

        # Creates a dictionary mapping headers (column names) to the animal-specific extracted values for these
        # headers. This procedure assumes that the headers are contiguous, start from column A, and the animal has data
        # for all or most present headers in the same sequential order as headers are encountered.
        animal_data: dict[str, Any] = {}
        for index, header in enumerate(self._headers):
            # Handles unlikely scenario of animal having more data than headers
            animal_data[header.lower()] = row_values[index] if index < len(row_values) else None

        # Parses the animal data and packages it into the SurgeryData instance:

        # Subject Data. We expect all subject data headers to always be present in all surgery sheets.
        subject_data = SubjectData(
            id=animal_data["id"],
            ear_punch=animal_data["ear punch"],
            sex=animal_data["sex"],
            genotype=animal_data["genotype"],
            date_of_birth_us=_convert_date_time_to_timestamp(date=animal_data["dob"], time="12:00"),
            weight_g=float(animal_data["weight (g)"]),
            cage=int(animal_data["cage #"]),
            location_housed=animal_data["location housed"],
            status=animal_data["status"],
        )

        # Procedure Data. Similar to subject data, we expect all required headers to be present for all procedures.
        procedure_data = ProcedureData(
            surgery_start_us=_convert_date_time_to_timestamp(date=animal_data["date"], time=animal_data["start"]),
            surgery_end_us=_convert_date_time_to_timestamp(date=animal_data["date"], time=animal_data["end"]),
            surgeon=animal_data["surgeon"],
            protocol=animal_data["protocol"],
            surgery_notes=animal_data["surgery notes"],
            post_op_notes=animal_data["post-op notes"],
            surgery_quality=animal_data["surgery quality"],
        )

        # Drug Data. The surgery log tracks each drug in a dedicated pair of volume and code columns. Drugs without a
        # recorded volume were not administered during the processed surgery and are excluded from the assembled list.
        # Since early surgery log versions did not use drug / injection / implant codes, code parsing has fall-back
        # default values (0). A code value of 0 should be interpreted as not having a code.
        drugs = []
        for drug_name, column_stem in _SURGERY_LOG_DRUGS.items():
            drug_volume = animal_data.get(f"{column_stem} (ml)")
            if drug_volume is not None:
                drugs.append(
                    DrugData(
                        drug=drug_name,
                        drug_volume_ml=drug_volume,
                        drug_code=str(animal_data.get(f"{column_stem} code", 0)),
                    )
                )

        # Determines the number of implants and injections performed during the processed surgery. This is based on the
        # assumption that all implants and injections are named like 'Implant1', 'Injection2_location', etc.

        # Pre-creates the lists to store the digits associated with each implant and injection.
        implant_numbers = []
        injection_numbers = []

        # Loops over all available headers and determines which implant(s) and injection(s) were performed.
        for key in animal_data:
            # This extraction only considers the 'main' column that stores the name of each implant and injection.
            # Such columns do not contain the whitespace separators between multiple words.
            if " " not in key.strip() and ("implant" in key or "injection" in key):
                # Parses the first run of digits in the key as the implant or injection number.
                match = _DIGIT_PATTERN.search(key)
                if match is not None:
                    number = int(match.group())
                    if "implant" in key:
                        implant_numbers.append(number)
                    else:  # If the key is not 'implant', it must be an injection.
                        injection_numbers.append(number)

        # Extracts and packages the data for each implant into an ImplantData class instance. If the processed surgery
        # did not use any implants, no ImplantData instances will be created. This is determined either by the
        # placeholder list being empty or all implant column values being None (empty).
        implants = []
        for number in implant_numbers:
            # Builds the 'base' implant name from the number, then reads the name stored in the 'main' implant column.
            base_key = f"implant{number}"
            implant_name: str | None = animal_data.get(base_key)

            # If the implant name is 'None', the processed subject does not have this implant, despite the
            # header being present. If the name is a string, it processes the rest of the data
            if implant_name is not None:
                # Some surgeries (training ones) do not make use of stereotactic coordinates. In such cases, defaults
                # to a set of zeroes to indicate no valid coordinates to parse.
                ap, ml, dv = 0.0, 0.0, 0.0

                # If a valid coordinate string is found, parses ap, ml, and dv coordinates from the string.
                implant_coordinate_string: str | None = animal_data.get(f"{base_key} coordinates")
                if implant_coordinate_string is not None:
                    ap, ml, dv = _parse_stereotactic_coordinates(implant_coordinate_string)

                # Packages the data into an ImplantData class and appends it to the storage list.
                implants.append(
                    ImplantData(
                        implant=implant_name,
                        implant_target=animal_data[f"{base_key} location"],
                        implant_code=str(animal_data.get(f"{base_key} code", 0)),
                        implant_ap_coordinate_mm=ap,
                        implant_ml_coordinate_mm=ml,
                        implant_dv_coordinate_mm=dv,
                    )
                )

        # Same as implants, but parses injection data. The only minor difference is that InjectionData has an additional
        # field to store injection volume in nanoliters.
        injections = []
        for number in injection_numbers:
            base_key = f"injection{number}"
            injection_name: str | None = animal_data.get(base_key)

            if injection_name is not None:
                ap, ml, dv = 0.0, 0.0, 0.0

                injection_coordinate_string: str | None = animal_data.get(f"{base_key} coordinates")
                if injection_coordinate_string is not None:
                    ap, ml, dv = _parse_stereotactic_coordinates(injection_coordinate_string)

                injections.append(
                    InjectionData(
                        injection=injection_name,
                        injection_target=animal_data[f"{base_key} location"],
                        injection_volume_nl=animal_data[f"{base_key} volume (nl)"],
                        injection_code=str(animal_data.get(f"{base_key} code", 0)),
                        injection_ap_coordinate_mm=ap,
                        injection_ml_coordinate_mm=ml,
                        injection_dv_coordinate_mm=dv,
                    )
                )

        # Aggregates all data into a SurgeryData instance and returns it to caller
        return SurgeryData(
            subject=subject_data, procedure=procedure_data, drugs=drugs, implants=implants, injections=injections
        )

    def update_surgery_quality(self, quality: int) -> None:
        """Updates the processed animal's surgery quality value to the input value.

        Notes:
            In addition to writing the value, this method issues a second batchUpdate request that applies CENTER
            horizontal and MIDDLE vertical cell alignment formatting to the updated cell.

        Args:
            quality: The integer value that reflects the quality of the animal's surgical intervention for scientific
                data acquisition on a scale from 0 (unusable) to 3 (high-grade scientific publication). The 0-3 scale is
                advisory; this method does not validate that the input falls within that range.
        """
        quality_column = self._get_column_id("surgery quality")

        animal_index = self._animals.index(str(self._animal_id).zfill(5))
        row_number = animal_index + 2  # +2 to account for header row and 0-indexing

        cell_range = f"{quality_column}{row_number}"
        body = {"values": [[quality]]}
        self._service.spreadsheets().values().update(  # type: ignore[attr-defined]
            spreadsheetId=self._sheet_id,
            range=f"'{self._project_name}'!{cell_range}",
            valueInputOption="USER_ENTERED",
            body=body,
        ).execute(num_retries=_GOOGLE_API_MAX_RETRIES)

        # Transforms the column letter and the row index to the format necessary to apply formatting to the newly
        # written value.
        column_index = 0
        for char in quality_column.upper():  # type: ignore[union-attr]
            column_index = column_index * 26 + (ord(char) - ord("A") + 1)
        column_index -= 1  # Convert to 0-based index
        row_index_zero_based = row_number - 1

        # Gets the sheet ID for the project tab
        sheet_metadata = (
            self._service.spreadsheets()  # type: ignore[attr-defined]
            .get(spreadsheetId=self._sheet_id)
            .execute(num_retries=_GOOGLE_API_MAX_RETRIES)
        )
        sheet_id = None
        for sheet in sheet_metadata.get("sheets", []):
            if sheet["properties"]["title"] == self._project_name:
                sheet_id = sheet["properties"]["sheetId"]
                break

        if sheet_id is not None:
            # Applies center alignment formatting to the cell
            requests = [
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": row_index_zero_based,
                            "endRowIndex": row_index_zero_based + 1,
                            "startColumnIndex": column_index,
                            "endColumnIndex": column_index + 1,
                        },
                        "cell": {"userEnteredFormat": {"horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}},
                        "fields": "userEnteredFormat.horizontalAlignment,userEnteredFormat.verticalAlignment",
                    }
                }
            ]
            self._service.spreadsheets().batchUpdate(  # type: ignore[attr-defined]
                spreadsheetId=self._sheet_id,
                body={"requests": requests},
            ).execute(num_retries=_GOOGLE_API_MAX_RETRIES)

    def _get_column_id(self, column_name: str) -> str | None:
        """Returns the surgery log column ID (letter) for the given column name.

        Args:
            column_name: The name of the target column as it appears in the header row.

        Returns:
            The column ID (e.g., "A", "B", "C") corresponding to the target column name. If the target column header
            does not exist, the method returns None to indicate the header is not available.
        """
        if column_name.lower() in self._headers:
            return self._headers[column_name]
        return None


class WaterLog:
    """Interfaces with the Sollertia platform's water restriction and animal interaction log Google Sheet.

    Notes:
        This class is purpose-built to work with the specific water restriction and animal interaction log format used
        by the Sollertia platform.

        This class is specifically designed to access the log data in a 'session-oriented' format. Specifically, it
        assumes that the log is accessed as part of a data acquisition session that uses a specific row to store all
        session-specific data.

    Args:
        animal_id: The unique identifier of the animal for which to access the log data.
        session_date: The date of the data acquisition session for which to access the log data, in the format
            YYYY-MM-DD-HH-MM-SS-US.
        credentials_path: The path to the .JSON file containing the service account credentials for accessing the
            water restriction and animal interaction log.
        sheet_id: The unique identifier of the Google Sheet that contains the water restriction and animal interaction
            data.

    Attributes:
        _sheet_id: The unique identifier of the water restriction and animal interaction log Google Sheet.
        _service: The API service instance that bidirectionally interfaces with the log.
        _animals: Stores the unique identifiers of all animals whose data is stored in the log.
        _headers: Maps the water restriction and animal interaction log headers (column names) to the Excel-style
            Google Sheet column letters (A, B, etc.).
        _animal_tab_id: The unique identifier of the log's tab that stores the target animal's data.
        _current_time: The session start time converted to the host machine's local time, in the HH:MM format.
        _session_row_index: The index of the log's row that stores the processed session's data.

    Raises:
        ValueError: If the target Google Sheet is not a valid Sollertia platform water restriction and animal
            interaction log.
    """

    def __init__(
        self,
        animal_id: int,
        session_date: str,
        credentials_path: Path,
        sheet_id: str,
    ) -> None:
        # Saves ID data to be reused during later instance method calls.
        self._animal_id: int = animal_id
        self._sheet_id: str = sheet_id

        # Generates the credentials' object to access the target Google Sheet. In contrast to surgery data, this object
        # requires write access.
        credentials = Credentials.from_service_account_file(  # type: ignore[no-untyped-call]
            filename=str(credentials_path), scopes=("https://www.googleapis.com/auth/spreadsheets",)
        )

        # Uses the credentials' object to build the access service for the target Google Sheet. This service is then
        # used to write the sheet data via HTTP request(s). Disables the discovery-document cache, which is unsupported
        # with the installed oauth2client version and only emits a spurious warning when left enabled.
        self._service: Resource = build(
            serviceName="sheets", version="v4", credentials=credentials, cache_discovery=False
        )

        # Gets all tab names from the sheet metadata
        sheet_metadata = (
            self._service.spreadsheets().get(spreadsheetId=sheet_id).execute(num_retries=_GOOGLE_API_MAX_RETRIES)
        )
        tabs = sheet_metadata.get("sheets", [])

        # Filters for tabs with digit-only names and extract them as animal IDs. This relies on all water restriction
        # logs following the general lab design, where tabs are named after animal numeric IDs and each tab is used to
        # store the animals' data.
        animal_ids = []
        self._animal_tab_id: int = -1

        for tab in tabs:
            tab_name = tab["properties"]["title"]
            tab_id = tab["properties"]["sheetId"]

            if tab_name.isdigit():  # Checks if the tab name contains only digits
                animal_ids.append(tab_name)

                # If this tab matches the target animal ID, stores its numeric sheet ID to the attribute
                if int(tab_name) == self._animal_id:
                    self._animal_tab_id = tab_id

        # Formats the animal IDs with zero-padding for consistent sorting
        self._animals: tuple[str, ...] = tuple(str(animal_id).zfill(5) for animal_id in animal_ids)

        # If no IDs are extracted, raises an error. This usually indicates that the target Google Sheet is not formatted
        # appropriately.
        if not self._animals:
            message = (
                f"Unable to interface with the water restriction and animal interaction log file for the animal "
                f"{self._animal_id}. The target log file does not contain any tabs named after unique animal "
                f"identifiers."
            )
            console.error(message=message, error=ValueError)

        # Otherwise, if the target animal ID is not in the extracted IDs list, raises an error.
        if str(self._animal_id).zfill(5) not in self._animals:
            message = (
                f"Unable to interface with the water restriction and animal interaction log file for the "
                f"animal {self._animal_id}. The target log file only contains the tabs for the following animals: "
                f"{[int(animal) for animal in sorted(self._animals)]}."
            )
            console.error(message=message, error=ValueError)

        # Retrieves all values stored in the second row of the Google Sheet tab with the name that matches the target
        # animal ID. Note: this is in contrast to the Surgery data log, where the headers are stored in the first sheet
        # row.
        # Extracts the entire second row.
        headers = (
            self._service.spreadsheets()
            .values()
            .get(spreadsheetId=sheet_id, range=f"'{self._animal_id}'!2:2")
            .execute(num_retries=_GOOGLE_API_MAX_RETRIES)
        )

        header_values = headers.get("values", [[]])[0]
        if not header_values:
            message = (
                f"Unable to interface with the water restriction and animal interaction log file for the "
                f"animal {self._animal_id}. The second row of the target log file's tab does not contain the "
                f"expected column headers."
            )
            console.error(message=message, error=ValueError)

        self._headers: dict[str, str] = {}
        for index, header in enumerate(header_values):
            column_letter = _convert_index_to_column_letter(index)
            self._headers[str(header).strip().lower()] = column_letter

        missing_headers = [
            required_header
            for required_header in _REQUIRED_WATER_RESTRICTION_HEADERS
            if required_header.lower() not in self._headers
        ]

        if missing_headers:
            missing_headers_str = ", ".join(f"'{header}'" for header in sorted(missing_headers))
            message = (
                f"Unable to interface with the water restriction and animal interaction log file for the "
                f"animal {self._animal_id}. The following required headers are missing from the log file: "
                f"{missing_headers_str}."
            )
            console.error(message=message, error=ValueError)

        # Parses the session's UTC timestamp from its name and renders it in the host machine's local time, which is
        # the timezone the log files use for user convenience.
        local_datetime = parse_session_timestamp(session_name=session_date, utc_timezone=False)
        if local_datetime is None:
            message = (
                f"Unable to interface with the water restriction and animal interaction log file for the animal "
                f"{self._animal_id}. The session name '{session_date}' is not a valid Sollertia session timestamp."
            )
            console.error(message=message, error=ValueError)

        # Formats the date to appear the same way as used in the log file.
        formatted_date = local_datetime.strftime("%-m/%-d/%y")

        # Formats the session's start time in the same way as used in the log file and saves it to a class attribute.
        self._current_time: str = local_datetime.strftime("%H:%M")

        # Finds the row inside the water restriction log file with session's date. This assumes that the log file is
        # pre-filled with dates. If not, this method will enter a loop to prompt the user to resolve the date issue.
        self._session_row_index: int = self._find_date_row(formatted_date)

    def __del__(self) -> None:
        """Terminates the HTTP connection to the processed water restriction and animal interaction log when the
        instance is garbage-collected.
        """
        self._service.close()

    def update_water_log(self, weight: float, water_ml: float, experimenter_id: str, session_type: str) -> None:
        """Updates the processed data acquisition session's row in the processed log file with the input animal's data.

        Notes:
            For this method to work as intended, the processed log tab must be pre-filled to contain the processed
            session's date.

        Args:
            weight: The weight of the animal, in grams, at the onset of the data acquisition session.
            water_ml: The combined volume of water, in milliliters, given to the animal by the acquisition system
                during runtime and by the experimenter after the runtime.
            experimenter_id: The unique identifier of the experimenter supervising the data acquisition session.
            session_type: The type of the data acquisition session.
        """
        # Writes each value to the appropriate column, using the same formatting as used in row 3. Since the sheet
        # is checked for validity and the session row index is discovered at class instantiation, this is a fairly
        # simple writing procedure.
        row_index = self._session_row_index
        self._write_value(column_name="weight (g)", row_index=row_index, value=weight)
        self._write_value(column_name="given by:", row_index=row_index, value=experimenter_id)
        self._write_value(column_name="water given (mL)", row_index=row_index, value=water_ml)
        self._write_value(column_name="behavior", row_index=row_index, value=session_type)
        self._write_value(column_name="time", row_index=row_index, value=self._current_time)

    def _find_date_row(self, target_date: str) -> int:
        """Finds the processed log's row index associated with the target date.

        Args:
            target_date: The date to find, in '%-m/%-d/%y' format (e.g. '5/24/26').

        Returns:
            The row index (1-based), containing the target date's data.

        Raises:
            ValueError: If the target date is not found inside the processed log file.
        """
        # Gets the date column letter
        date_column = self._headers["date"]

        # Retrieves all dates from the date column (row 3 and below)
        date_data = (
            self._service.spreadsheets()  # type: ignore[attr-defined]
            .values()
            .get(spreadsheetId=self._sheet_id, range=f"'{self._animal_id}'!{date_column}3:{date_column}")
            .execute(num_retries=_GOOGLE_API_MAX_RETRIES)
        )
        date_values = date_data.get("values", [])

        for index, date_cell in enumerate(date_values):
            if date_cell and date_cell[0] == target_date:
                # Adds 3 to account for 0-indexing and the fact we started from row 3
                return index + 3
        message = (
            f"Unable to find the row for the target date {target_date} inside the water restriction and "
            f"animal interaction log file for the animal {self._animal_id}. Update the log to include the "
            f"specified date and rerun the command that caused this error."
        )
        console.error(message=message, error=ValueError)  # Aborts with an error
        # Unreachable: console.error() is NoReturn, but ruff cannot trace NoReturn through method calls (RET503).
        raise ValueError(message)  # pragma: no cover

    def _write_value(self, column_name: str, row_index: int, value: float | str) -> None:
        """Writes the input value to the target log's cell based on the column name and row index.

        Args:
            column_name: The name of the column to write to.
            row_index: The row index (1-based) to write to.
            value: The value to write.
        """
        column_letter = self._headers[column_name.lower()]
        cell_range = f"{column_letter}{row_index}"

        # Formats value based on its type and column
        formatted_value = value
        if column_name.lower() == "weight (g)" or column_name.lower() == "water given (ml)":
            formatted_value = round(float(value), ndigits=1)

        body = {"values": [[formatted_value]]}
        self._service.spreadsheets().values().update(  # type: ignore[attr-defined]
            spreadsheetId=self._sheet_id,
            range=f"'{self._animal_id}'!{cell_range}",
            valueInputOption="USER_ENTERED",
            body=body,
        ).execute(num_retries=_GOOGLE_API_MAX_RETRIES)

        # Transforms the column letter and the row index to the format necessary to apply formatting to the newly
        # written value.
        column_index = 0
        for char in column_letter.upper():
            column_index = column_index * 26 + (ord(char) - ord("A") + 1)
        column_index -= 1  # Converts to 0-based index
        row_index_zero_based = row_index - 1

        # Applies formatting to the newly written value using the cached sheet ID
        requests = [
            {
                "repeatCell": {
                    "range": {
                        "sheetId": self._animal_tab_id,
                        "startRowIndex": row_index_zero_based,
                        "endRowIndex": row_index_zero_based + 1,
                        "startColumnIndex": column_index,
                        "endColumnIndex": column_index + 1,
                    },
                    "cell": {"userEnteredFormat": {"horizontalAlignment": "CENTER", "verticalAlignment": "MIDDLE"}},
                    "fields": "userEnteredFormat.horizontalAlignment,userEnteredFormat.verticalAlignment",
                }
            }
        ]
        self._service.spreadsheets().batchUpdate(  # type: ignore[attr-defined]
            spreadsheetId=self._sheet_id,
            body={"requests": requests},
        ).execute(num_retries=_GOOGLE_API_MAX_RETRIES)
