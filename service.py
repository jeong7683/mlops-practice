import bentoml
import numpy as np
import numpy.typing as npt
import pandas as pd
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel


class Features(BaseModel):
    bhk: int
    size: int
    floor: str
    area_type: str
    city: str
    furnishing_status: str
    tenant_preferred: str
    bathroom: int
    point_of_contact: str


# 학습 코드에서 저장한 베스트 모델을 가져올 것 (house_rent:latest)
bento_ml = bentoml.sklearn.get("house_rent:latest")
model_runner = bento_ml.to_runner()

svc = bentoml.Service("rent_house_regressor", runners=[model_runner])


@svc.api(input=JSON(pydantic_model=Features), output=NumpyNdarray())
async def predict(input_data: Features) -> npt.NDArray:  ## 비동기 함수: async
    input_df = pd.DataFrame([input_data.dict()])
    log_pred = await model_runner.predict.async_run(
        input_df
    )  ## 비동기 실행: await, async_run
    return np.expm1(log_pred)  ## 로그변환 되어 있음 (지수변환 필요)
