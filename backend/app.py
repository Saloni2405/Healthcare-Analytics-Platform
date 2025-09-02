from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()

# allow frontend (react) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to ["http://localhost:3000", "http://localhost:3001"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "ok", "service": "Healthcare Analytics Platform"}

# new summary endpoint
@app.get("/summary")
def get_summary():
    try:
        # load dataset
        df = pd.read_csv("data/synthetic_healthcare_dataset.csv")

        # compute stats
        patient_count = len(df)
        avg_age = round(df["Age"].mean(), 2)
        avg_bed_occupancy = round(df["Bed_Occupancy_Rate"].mean(), 2)
        chronic_counts = df["Chronic_Condition"].value_counts().to_dict()

        return {
            "total_patients": patient_count,
            "average_age": avg_age,
            "average_bed_occupancy": avg_bed_occupancy,
            "chronic_condition_distribution": chronic_counts,
        }
    except Exception as e:
        return {"error": str(e)}
