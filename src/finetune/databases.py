"""Synthetic SQLite databases for training data.

None of these use the Chinook benchmark's schema or contents, so what the model learns here
can't leak benchmark answers. Each database has a different domain, so the model has to read
the schema it's given instead of memorizing table names.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Column:
    name: str
    kind: str  # id, name, category, number, money, date, fk
    choices: tuple = ()  # Category values, or the referenced table for a foreign key.


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    rows: int

    @property
    def id_column(self) -> str:
        return self.columns[0].name


@dataclass(frozen=True)
class Domain:
    name: str
    tables: tuple[Table, ...]
    # Terms for things this database doesn't hold, used for must-refuse questions.
    missing: tuple[str, ...]


def _c(name, kind, *choices):
    return Column(name, kind, tuple(choices))


DOMAINS = [
    Domain("hospital", (
        Table("Department", (_c("DepartmentId", "id"), _c("Name", "name")), 8),
        Table("Doctor", (_c("DoctorId", "id"), _c("FullName", "name"), _c("DepartmentId", "fk", "Department"),
                         _c("Grade", "category", "Junior", "Senior", "Consultant")), 40),
        Table("Patient", (_c("PatientId", "id"), _c("FullName", "name"), _c("Age", "number"),
                          _c("City", "category", "Leeds", "York", "Hull", "Derby", "Bath")), 300),
        Table("Visit", (_c("VisitId", "id"), _c("PatientId", "fk", "Patient"), _c("DoctorId", "fk", "Doctor"),
                        _c("VisitDate", "date"), _c("Fee", "money")), 1500),
    ), ("insurance claim amounts", "patient blood types", "staff salaries", "bed occupancy rates")),
    Domain("university", (
        Table("Faculty", (_c("FacultyId", "id"), _c("Name", "name")), 6),
        Table("Course", (_c("CourseId", "id"), _c("Title", "name"), _c("FacultyId", "fk", "Faculty"),
                         _c("Credits", "number")), 60),
        Table("Student", (_c("StudentId", "id"), _c("FullName", "name"), _c("EnrolYear", "number"),
                          _c("Campus", "category", "North", "South", "East")), 500),
        Table("Enrollment", (_c("EnrollmentId", "id"), _c("StudentId", "fk", "Student"), _c("CourseId", "fk", "Course"),
                             _c("Term", "category", "Autumn", "Winter", "Spring"), _c("Score", "number")), 2500),
    ), ("tuition payments", "student home addresses", "professor ratings", "library fines")),
    Domain("logistics", (
        Table("Warehouse", (_c("WarehouseId", "id"), _c("Name", "name"),
                            _c("Region", "category", "Pacific", "Central", "Atlantic")), 10),
        Table("Driver", (_c("DriverId", "id"), _c("FullName", "name"), _c("HomeWarehouseId", "fk", "Warehouse")), 50),
        Table("Shipment", (_c("ShipmentId", "id"), _c("DriverId", "fk", "Driver"), _c("OriginId", "fk", "Warehouse"),
                           _c("ShipDate", "date"), _c("WeightKg", "number"), _c("Charge", "money"),
                           _c("Status", "category", "Delivered", "Delayed", "Lost")), 2000),
    ), ("fuel costs", "driver hourly wages", "truck maintenance records", "customer satisfaction scores")),
    Domain("retail", (
        Table("Brand", (_c("BrandId", "id"), _c("Name", "name")), 15),
        Table("Item", (_c("ItemId", "id"), _c("Title", "name"), _c("BrandId", "fk", "Brand"), _c("ListPrice", "money"),
                       _c("Family", "category", "Home", "Garden", "Kitchen", "Outdoor")), 200),
        Table("Store", (_c("StoreId", "id"), _c("Name", "name"),
                        _c("State", "category", "Ohio", "Iowa", "Utah", "Texas")), 20),
        Table("Sale", (_c("SaleId", "id"), _c("ItemId", "fk", "Item"), _c("StoreId", "fk", "Store"),
                       _c("SaleDate", "date"), _c("Units", "number"), _c("Amount", "money")), 3000),
    ), ("supplier costs", "advertising budgets", "employee turnover", "return reasons")),
    Domain("airline", (
        Table("Airport", (_c("AirportId", "id"), _c("Name", "name"),
                          _c("Country", "category", "Peru", "Chile", "Spain", "Kenya", "Japan")), 25),
        Table("Flight", (_c("FlightId", "id"), _c("OriginId", "fk", "Airport"), _c("DestinationId", "fk", "Airport"),
                         _c("Minutes", "number")), 200),
        Table("Passenger", (_c("PassengerId", "id"), _c("FullName", "name"), _c("Tier", "category", "Basic", "Silver", "Gold")), 600),
        Table("Booking", (_c("BookingId", "id"), _c("FlightId", "fk", "Flight"), _c("PassengerId", "fk", "Passenger"),
                          _c("TravelDate", "date"), _c("Fare", "money")), 3000),
    ), ("jet fuel prices", "pilot schedules", "baggage claim times", "in-flight meal choices")),
    Domain("energy", (
        Table("Site", (_c("SiteId", "id"), _c("Name", "name"),
                       _c("Source", "category", "Solar", "Wind", "Hydro"), _c("CapacityMw", "number")), 30),
        Table("Operator", (_c("OperatorId", "id"), _c("FullName", "name"), _c("SiteId", "fk", "Site")), 60),
        Table("Reading", (_c("ReadingId", "id"), _c("SiteId", "fk", "Site"), _c("ReadingDate", "date"),
                          _c("OutputMwh", "number"), _c("Revenue", "money")), 4000),
    ), ("carbon credit prices", "equipment warranty dates", "grid outage counts", "operator certifications")),
    Domain("library", (
        Table("Author", (_c("AuthorId", "id"), _c("FullName", "name"), _c("Nation", "category", "Ghana", "Norway", "Brazil", "India")), 80),
        Table("Book", (_c("BookId", "id"), _c("Title", "name"), _c("AuthorId", "fk", "Author"), _c("Pages", "number"),
                       _c("Subject", "category", "History", "Science", "Poetry", "Travel")), 500),
        Table("Member", (_c("MemberId", "id"), _c("FullName", "name"), _c("JoinYear", "number")), 400),
        Table("Loan", (_c("LoanId", "id"), _c("BookId", "fk", "Book"), _c("MemberId", "fk", "Member"),
                       _c("LoanDate", "date"), _c("LateFee", "money")), 3000),
    ), ("book purchase prices", "member email addresses", "shelf locations", "reading group schedules")),
    Domain("gym", (
        Table("Club", (_c("ClubId", "id"), _c("Name", "name"), _c("District", "category", "Riverside", "Harbor", "Uptown")), 12),
        Table("Trainer", (_c("TrainerId", "id"), _c("FullName", "name"), _c("ClubId", "fk", "Club")), 45),
        Table("Athlete", (_c("AthleteId", "id"), _c("FullName", "name"), _c("Level", "category", "Starter", "Regular", "Elite")), 350),
        Table("Session", (_c("SessionId", "id"), _c("TrainerId", "fk", "Trainer"), _c("AthleteId", "fk", "Athlete"),
                          _c("SessionDate", "date"), _c("Minutes", "number"), _c("Price", "money")), 2500),
    ), ("equipment repair costs", "athlete body weights", "trainer certifications", "membership cancellations")),
]

# Whole domains are held out, so validation shows whether the model handles unseen schemas.
VALIDATION_DOMAINS = ("library", "gym")

_FIRST = ("Ana Ben Chloe Dev Eli Fay Gus Hana Ivan Jo Kira Liam Mei Nia Omar Pia Quin Rosa Sam Tess Uma Vik "
          "Wes Xia Yan Zoe").split()
_LAST = ("Abara Bell Costa Diaz Egan Fox Gray Hale Ito Jain Kerr Lund Moss Nair Ochoa Park Quill Reyes "
         "Stone Tran Ueda Voss Wolfe Young Zhao").split()
_WORDS = ("Amber Birch Cedar Delta Ember Fern Grove Harbor Indigo Juniper Kestrel Linden Maple Nimbus Onyx "
          "Pine Quartz Raven Sable Tidal Umber Violet Willow Yarrow Zephyr").split()
_NOUNS = "Point Works Hall Center Lab Group House Union Studio Field".split()


def _name(kind_name: str, rng: random.Random, row: int) -> str:
    if "FullName" in kind_name:
        return f"{rng.choice(_FIRST)} {rng.choice(_LAST)} {row}"
    return f"{rng.choice(_WORDS)} {rng.choice(_NOUNS)} {row}"


def _value(column: Column, rng: random.Random, row: int, existing: dict[str, int]):
    if column.kind == "id":
        return row
    if column.kind == "name":
        return _name(column.name, rng, row)
    if column.kind == "category":
        return rng.choice(column.choices)
    if column.kind == "fk":
        # Skewed, so rankings have clear winners and questions rarely tie.
        count = existing[column.choices[0]]
        return min(int(rng.paretovariate(1.2)), count)
    if column.kind == "number":
        return rng.randint(1, 500)
    if column.kind == "money":
        return round(rng.uniform(1, 400), 2)
    if column.kind == "date":
        return f"{rng.randint(2019, 2024)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    raise ValueError(column.kind)


def _sql_type(column: Column) -> str:
    if column.kind in {"id", "fk", "number"}:
        return "INTEGER"
    if column.kind == "money":
        return "NUMERIC(10,2)"
    return "TEXT"


def build_database(domain: Domain, directory: str | Path, seed: int = 0) -> Path:
    path = Path(directory) / f"{domain.name}.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    rng = random.Random(f"{domain.name}:{seed}")
    conn = sqlite3.connect(path)
    existing: dict[str, int] = {}
    for table in domain.tables:
        definitions = []
        keys = []
        for column in table.columns:
            suffix = " PRIMARY KEY" if column.kind == "id" else ""
            definitions.append(f'"{column.name}" {_sql_type(column)}{suffix}')
            if column.kind == "fk":
                keys.append(f'FOREIGN KEY ("{column.name}") REFERENCES "{column.choices[0]}" '
                            f'("{_id_of(domain, column.choices[0])}")')
        conn.execute(f'CREATE TABLE "{table.name}" ({", ".join(definitions + keys)})')
        rows = [
            tuple(_value(column, rng, row, existing) for column in table.columns)
            for row in range(1, table.rows + 1)
        ]
        marks = ",".join("?" * len(table.columns))
        conn.executemany(f'INSERT INTO "{table.name}" VALUES ({marks})', rows)
        existing[table.name] = table.rows
    conn.commit()
    conn.close()
    return path


def _id_of(domain: Domain, table_name: str) -> str:
    return next(t for t in domain.tables if t.name == table_name).id_column
