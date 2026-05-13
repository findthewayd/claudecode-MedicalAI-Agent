import pickle
import os

HISTORY_FILE = "db/patient_history.pkl"

class PatientMemory:
    def __init__(self):
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "rb") as f:
                self.history = pickle.load(f)
        else:
            self.history = {}

    def get_history(self, patient_id):
        return self.history.get(patient_id, [])

    def add_message(self, patient_id, role, content):
        if patient_id not in self.history:
            self.history[patient_id] = []
        self.history[patient_id].append({"role": role, "content": content})
        self.save()

    def save(self):
        os.makedirs("db", exist_ok=True)
        with open(HISTORY_FILE, "wb") as f:
            pickle.dump(self.history, f)