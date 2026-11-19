from typing import List, Literal, Optional, Union
from pydantic import BaseModel, ConfigDict, Field


class ClientCoreFinancials(BaseModel):
    AMT_INCOME_TOTAL: Optional[float] = Field(default=None, gt=0, description="Total annual income of the client")
    AMT_CREDIT: Optional[float] = Field(default=None, gt=0, description="Credit amount of the loan")
    AMT_ANNUITY: Optional[float] = Field(default=None, gt=0, description="Loan annuity")
    AMT_GOODS_PRICE: Optional[float] = Field(default=None, gt=0, description="Price of goods for consumer loans")

    EXT_SOURCE_1: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized external score 1")
    EXT_SOURCE_2: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized external score 2")
    EXT_SOURCE_3: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Normalized external score 3")

    AMT_REQ_CREDIT_BUREAU_HOUR: Optional[float] = Field(default=None, ge=0)
    AMT_REQ_CREDIT_BUREAU_DAY: Optional[float] = Field(default=None, ge=0)
    AMT_REQ_CREDIT_BUREAU_WEEK: Optional[float] = Field(default=None, ge=0)
    AMT_REQ_CREDIT_BUREAU_MON: Optional[float] = Field(default=None, ge=0)
    AMT_REQ_CREDIT_BUREAU_QRT: Optional[float] = Field(default=None, ge=0)
    AMT_REQ_CREDIT_BUREAU_YEAR: Optional[float] = Field(default=None, ge=0)


class ClientDemographicsAndCategories(BaseModel):
    NAME_CONTRACT_TYPE: Optional[Literal["Cash loans", "Revolving loans"]] = None
    CODE_GENDER: Optional[Literal["M", "F", "XNA"]] = None
    FLAG_OWN_CAR: Optional[Literal["Y", "N"]] = None
    FLAG_OWN_REALTY: Optional[Literal["Y", "N"]] = None

    CNT_CHILDREN: Optional[int] = Field(default=None, ge=0, le=30)
    CNT_FAM_MEMBERS: Optional[float] = Field(default=None, ge=1.0, le=30.0)
    OWN_CAR_AGE: Optional[float] = Field(default=None, ge=0.0, le=100.0)

    DAYS_BIRTH: Optional[int] = Field(default=None, le=0, description="Client age in days (negative)")
    DAYS_EMPLOYED: Optional[int] = Field(default=None, description="Days employed (365243 if unemployed)")
    DAYS_REGISTRATION: Optional[float] = Field(default=None, le=0)
    DAYS_ID_PUBLISH: Optional[int] = Field(default=None, le=0)
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(default=None, le=0)

    NAME_TYPE_SUITE: Optional[str] = None
    NAME_INCOME_TYPE: Optional[str] = None
    NAME_EDUCATION_TYPE: Optional[str] = None
    NAME_FAMILY_STATUS: Optional[str] = None
    NAME_HOUSING_TYPE: Optional[str] = None
    OCCUPATION_TYPE: Optional[str] = None
    WEEKDAY_APPR_PROCESS_START: Optional[
        Literal["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY"]
    ] = None
    HOUR_APPR_PROCESS_START: Optional[int] = Field(default=None, ge=0, le=23)
    ORGANIZATION_TYPE: Optional[str] = None


class ClientRegionAndSocial(BaseModel):
    REGION_POPULATION_RELATIVE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    REGION_RATING_CLIENT: Optional[int] = Field(default=None, ge=1, le=3)
    REGION_RATING_CLIENT_W_CITY: Optional[int] = Field(default=None, ge=1, le=3)

    REG_REGION_NOT_LIVE_REGION: Optional[Literal[0, 1]] = None
    REG_REGION_NOT_WORK_REGION: Optional[Literal[0, 1]] = None
    LIVE_REGION_NOT_WORK_REGION: Optional[Literal[0, 1]] = None
    REG_CITY_NOT_LIVE_CITY: Optional[Literal[0, 1]] = None
    REG_CITY_NOT_WORK_CITY: Optional[Literal[0, 1]] = None
    LIVE_CITY_NOT_WORK_CITY: Optional[Literal[0, 1]] = None

    OBS_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, ge=0)
    DEF_30_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, ge=0)
    OBS_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, ge=0)
    DEF_60_CNT_SOCIAL_CIRCLE: Optional[float] = Field(default=None, ge=0)


class ClientHousingMetrics(BaseModel):
    # Normalized building & apartment metrics (AVG)
    APARTMENTS_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    BASEMENTAREA_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    YEARS_BEGINEXPLUATATION_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    YEARS_BUILD_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    COMMONAREA_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ELEVATORS_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ENTRANCES_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    FLOORSMAX_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    FLOORSMIN_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    LANDAREA_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    LIVINGAPARTMENTS_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    LIVINGAREA_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    NONLIVINGAPARTMENTS_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    NONLIVINGAREA_AVG: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Normalized building & apartment metrics (MODE)
    APARTMENTS_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    BASEMENTAREA_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    YEARS_BEGINEXPLUATATION_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    YEARS_BUILD_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    COMMONAREA_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ELEVATORS_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ENTRANCES_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    FLOORSMAX_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    FLOORSMIN_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    LANDAREA_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    LIVINGAPARTMENTS_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    LIVINGAREA_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    NONLIVINGAPARTMENTS_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    NONLIVINGAREA_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Normalized building & apartment metrics (MEDI)
    APARTMENTS_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    BASEMENTAREA_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    YEARS_BEGINEXPLUATATION_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    YEARS_BUILD_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    COMMONAREA_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ELEVATORS_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    ENTRANCES_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    FLOORSMAX_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    FLOORSMIN_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    LANDAREA_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    LIVINGAPARTMENTS_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    LIVINGAREA_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    NONLIVINGAPARTMENTS_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    NONLIVINGAREA_MEDI: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    TOTALAREA_MODE: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    FONDKAPREMONT_MODE: Optional[str] = None
    HOUSETYPE_MODE: Optional[str] = None
    WALLSMATERIAL_MODE: Optional[str] = None
    EMERGENCYSTATE_MODE: Optional[Literal["Yes", "No"]] = None


class ClientFlagsAndDocuments(BaseModel):
    FLAG_MOBIL: Optional[Literal[0, 1]] = None
    FLAG_EMP_PHONE: Optional[Literal[0, 1]] = None
    FLAG_WORK_PHONE: Optional[Literal[0, 1]] = None
    FLAG_CONT_MOBILE: Optional[Literal[0, 1]] = None
    FLAG_PHONE: Optional[Literal[0, 1]] = None
    FLAG_EMAIL: Optional[Literal[0, 1]] = None

    FLAG_DOCUMENT_2: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_3: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_4: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_5: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_6: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_7: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_8: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_9: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_10: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_11: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_12: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_13: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_14: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_15: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_16: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_17: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_18: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_19: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_20: Optional[Literal[0, 1]] = None
    FLAG_DOCUMENT_21: Optional[Literal[0, 1]] = None


class ClientApplication(
    ClientCoreFinancials,
    ClientDemographicsAndCategories,
    ClientRegionAndSocial,
    ClientHousingMetrics,
    ClientFlagsAndDocuments,
):
    model_config = ConfigDict(extra="forbid")

    SK_ID_CURR: int = Field(..., gt=0, description="Unique ID of current loan application")


class ClientRequest(BaseModel):
    data: Union[ClientApplication, List[ClientApplication]] = Field(
        ..., description="Single client application or list of client applications"
    )