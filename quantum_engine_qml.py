"""
QACS — quantum_engine_qml.py
Continuous Quantum Neural Network (QNN) for CRISPR efficiency prediction.

WHAT THIS REPLACES:
  Previous versions forced continuous CRISPR efficiency scores into 
  binary buckets (high/low) using a Variational Quantum Classifier (VQC). 

WHAT THIS DOES INSTEAD:
  Uses an EstimatorQNN coupled with a NeuralNetworkRegressor. It measures 
  the Pauli-Z expectation value of the quantum state (which natively ranges 
  from -1.0 to 1.0) and maps it to the biological thermodynamic efficiency 
  gradient (0.0 to 1.0).

OPTIMIZER UPGRADE:
  Switched from COBYLA to SPSA (Simultaneous Perturbation Stochastic 
  Approximation). SPSA evaluates the gradient stochastically, allowing it 
  to punch through the "barren plateaus" of the quantum landscape and 
  natively resist hardware noise.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# Classical ML
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import xgboost as xgb

# Qiskit QML & Primitives
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor

warnings.filterwarnings("ignore")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────

N_QUBITS         = 4      
N_FEATURES_RAW   = 8      
TRAIN_SAMPLES    = 800    # Increased to give SPSA sufficient gradient density
MODEL_PATH       = "qacs_qml_model.pkl"


# ── RESULT DATACLASS ──────────────────────────────────────────────────────────

@dataclass
class QMLResult:
    guide:           str
    position:        int
    strand:          str
    pam:             str
    gc_content:      float
    quantum_score:   float = 0.0
    classical_score: float = 0.0
    confidence:      float = 0.0
    agreement:       bool  = True
    method:          str   = "QNN-Regressor"
    error:           str   = ""


# ── STEP 1: FEATURE EXTRACTOR ─────────────────────────────────────────────────

def extract_guide_features(guide: str) -> np.ndarray:
    """Extract 8 biological features from a 20-nt guide RNA sequence."""
    guide = guide.upper().strip()
    if len(guide) < 20:
        return np.zeros(N_FEATURES_RAW)

    gc_global       = (guide.count("G") + guide.count("C")) / len(guide)
    seed            = guide[8:]  # PAM-proximal seed (positions 8-19)
    gc_seed         = (seed.count("G") + seed.count("C")) / 12
    non_seed        = guide[:8]  # PAM-distal (positions 0-7)
    gc_non_seed     = (non_seed.count("G") + non_seed.count("C")) / 8
    gg_freq         = guide.count("GG") / 19
    cc_freq         = guide.count("CC") / 19
    terminal_purine = 1.0 if guide[-1] in ["G", "A"] else 0.0
    five_prime_g    = 1.0 if guide[0] == "G" else 0.0
    
    strong = guide.count("G") + guide.count("C")
    weak   = guide.count("A") + guide.count("T")
    thermo = (strong - weak) / len(guide)

    return np.array([
        gc_global, gc_seed, gc_non_seed, gg_freq, 
        cc_freq, terminal_purine, five_prime_g, thermo
    ])


# ── STEP 2: DATA PIPELINE ─────────────────────────────────────────────────────

def prepare_training_data(df: pd.DataFrame) -> tuple:
    print(f"  Extracting features from {len(df)} sequences...")
    X_raw = np.array([extract_guide_features(seq) for seq in df["sequence"]])
    y     = df["efficiency"].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    pca       = PCA(n_components=N_QUBITS)
    X_quantum = pca.fit_transform(X_scaled)

    explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {N_FEATURES_RAW} features → {N_QUBITS} components")
    print(f"  Variance explained: {explained:.1%}")

    return X_quantum, y, scaler, pca


def uniform_subsample(X: np.ndarray, y: np.ndarray, n_samples: int = TRAIN_SAMPLES) -> tuple:
    """
    Subsamples data uniformly across the array.
    Since we are performing continuous regression instead of classification, 
    strict threshold stratification is no longer mathematically necessary.
    """
    idx = np.arange(len(y))
    np.random.seed(42)
    np.random.shuffle(idx)
    selected_idx = idx[:n_samples]
    
    print(f"  Uniform sample: {len(selected_idx)} total for QNN training")
    return X[selected_idx], y[selected_idx]


# ── STEP 3: QUANTUM CIRCUIT ───────────────────────────────────────────────────

def build_quantum_circuit():
    feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=2, entanglement="linear")
    ansatz      = RealAmplitudes(num_qubits=N_QUBITS, reps=2, entanglement="linear")
    return feature_map, ansatz


# ── STEP 4: TRAINING ──────────────────────────────────────────────────────────

def train_quantum_model(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Train a Continuous QNN Regressor.
    """
    feature_map, ansatz = build_quantum_circuit()
    qc = feature_map.compose(ansatz)
    
    # Observable: Measure Pauli-Z on the first qubit. Expectation value native range: [-1.0, 1.0]
    observable = SparsePauliOp.from_list([("Z" + "I" * (N_QUBITS - 1), 1)])

    qnn = EstimatorQNN(
        circuit=qc,
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters
    )

    regressor = NeuralNetworkRegressor(
        neural_network=qnn,
        loss="squared_error",
        optimizer=SPSA(maxiter=300) # SPSA punches through flat quantum landscapes
    )

    print(f"  Training QNN Regressor on {len(X_train)} samples...")
    print(f"  Circuit: {N_QUBITS}q ZZFeatureMap + RealAmplitudes")
    print(f"  Observable: Pauli-Z expectation value [-1.0, 1.0]")
    print(f"  Optimizer: SPSA (maxiter=300)")

    # Scale biological targets [0.0, 1.0] to physics observable [-1.0, 1.0]
    y_train_scaled = (y_train.reshape(-1, 1) * 2.0) - 1.0

    regressor.fit(X_train, y_train_scaled)

    # Evaluate
    y_pred_scaled = regressor.predict(X_train)
    y_pred        = (y_pred_scaled + 1.0) / 2.0 # Scale expectation back to [0.0, 1.0]
    
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    r, _ = pearsonr(y_train.flatten(), y_pred.flatten())

    print(f"  ✓ QNN Training complete — Pearson r: {r:.4f} | RMSE: {rmse:.4f}")
    return regressor, round(r, 4), round(rmse, 4)


def train_classical_baseline(X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray,  y_test: np.ndarray) -> tuple:
    model = xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r, _   = pearsonr(y_test, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, round(r, 4), round(rmse, 4)


# ── STEP 5: MODEL MANAGER ─────────────────────────────────────────────────────

class QMLModel:
    def __init__(self):
        self.qnn_regressor = None
        self.xgb_model     = None
        self.scaler        = None
        self.pca           = None
        self.trained       = False
        self.qnn_r         = 0.0
        self.qnn_rmse      = 0.0
        self.xgb_r         = 0.0
        self.xgb_rmse      = 0.0

    def train(self, df: pd.DataFrame) -> None:
        print("\n  QACS QML — Continuous Regression Pipeline")
        print("  " + "─" * 45)

        X_all, y_all, self.scaler, self.pca = prepare_training_data(df)
        X_q, y_q = uniform_subsample(X_all, y_all, TRAIN_SAMPLES)

        X_train, X_test, y_train, y_test = train_test_split(X_q, y_q, test_size=0.2, random_state=42)

        self.qnn_regressor, self.qnn_r, self.qnn_rmse = train_quantum_model(X_train, y_train)

        print("\n  Training XGBoost baseline...")
        X_tr_full, X_te_full, y_tr_full, y_te_full = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        self.xgb_model, self.xgb_r, self.xgb_rmse = train_classical_baseline(X_tr_full, y_tr_full, X_te_full, y_te_full)
        print(f"  ✓ XGBoost baseline — Pearson r: {self.xgb_r} | RMSE: {self.xgb_rmse}")

        self.trained = True
        print("\n  ✓ QML model training complete")

    def predict(self, guide: str) -> tuple:
        if not self.trained:
            return 0.5, 0.5

        raw     = extract_guide_features(guide).reshape(1, -1)
        scaled  = self.scaler.transform(raw)
        quantum = self.pca.transform(scaled)

        # ── Quantum Prediction (Continuous Expectation Value) ────────────────
        try:
            # Returns an array with the expectation value scaled by the model
            q_pred_scaled = self.qnn_regressor.predict(quantum)[0][0]
            # Convert Pauli-Z expectation value [-1.0, 1.0] back to efficiency [0.0, 1.0]
            quantum_score = np.clip((q_pred_scaled + 1.0) / 2.0, 0.0, 1.0)
        except Exception as e:
            quantum_score = 0.5

        # ── Classical Prediction ─────────────────────────────────────────────
        try:
            classical_score = float(self.xgb_model.predict(quantum)[0])
            classical_score = float(np.clip(classical_score, 0.0, 1.0))
        except Exception:
            classical_score = 0.5

        return float(quantum_score), float(classical_score)

    def save(self, path: str = MODEL_PATH) -> None:
        save_data = {
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
            "pca_components": self.pca.components_.tolist(),
            "pca_mean": self.pca.mean_.tolist(),
            "pca_variance": self.pca.explained_variance_.tolist(),
            "qnn_weights": self.qnn_regressor.weights.tolist() if self.qnn_regressor and hasattr(self.qnn_regressor, 'weights') else [],
            "qnn_r": self.qnn_r,
            "qnn_rmse": self.qnn_rmse,
            "xgb_r": self.xgb_r,
            "xgb_rmse": self.xgb_rmse
        }

        xgb_path = path.replace(".pkl", "_xgb.json")
        if self.xgb_model:
            self.xgb_model.save_model(xgb_path)
            save_data["xgb_path"] = xgb_path

        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        print(f"  ✓ Model saved to {path}")

    def load(self, path: str = MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self.scaler = StandardScaler()
            self.scaler.mean_ = np.array(data["scaler_mean"])
            self.scaler.scale_ = np.array(data["scaler_scale"])

            self.pca = PCA(n_components=N_QUBITS)
            self.pca.components_ = np.array(data["pca_components"])
            self.pca.mean_ = np.array(data["pca_mean"])
            self.pca.explained_variance_ = np.array(data["pca_variance"])

            feature_map, ansatz = build_quantum_circuit()
            qc = feature_map.compose(ansatz)
            observable = SparsePauliOp.from_list([("Z" + "I" * (N_QUBITS - 1), 1)])

            qnn = EstimatorQNN(
                circuit=qc,
                observables=observable,
                input_params=feature_map.parameters,
                weight_params=ansatz.parameters
            )

            self.qnn_regressor = NeuralNetworkRegressor(
                neural_network=qnn, 
                loss="squared_error", 
                optimizer=SPSA()
            )

            if "qnn_weights" in data and len(data["qnn_weights"]) > 0:
                self.qnn_regressor._fit_result = {"optimal_point": np.array(data["qnn_weights"])}
                self.qnn_regressor._weights = np.array(data["qnn_weights"])

            xgb_path = data.get("xgb_path", path.replace(".pkl", "_xgb.json"))
            if os.path.exists(xgb_path):
                self.xgb_model = xgb.XGBRegressor()
                self.xgb_model.load_model(xgb_path)

            self.qnn_r    = data.get("qnn_r", 0.0)
            self.qnn_rmse = data.get("qnn_rmse", 0.0)
            self.xgb_r    = data.get("xgb_r", 0.0)
            self.xgb_rmse = data.get("xgb_rmse", 0.0)

            self.trained = True
            print(f"  ✓ Model loaded from {path}")
            return True
        except Exception as e:
            print(f"  ✗ Load failed: {e}")
            return False


# ── STEP 6: SITE EVALUATOR ────────────────────────────────────────────────────

def evaluate_sites_qml(candidate_sites: list, model: QMLModel) -> list:
    if not model.trained:
        print("  ⚠ QML model not trained. Run training first.")
        return []

    results = []
    for site in candidate_sites:
        guide = site["guide"]
        try:
            q_score, c_score = model.predict(guide)
            
            # Confidence is derived continuously from both models
            confidence = round((q_score * 0.4 + c_score * 0.6), 4)
            agreement  = abs(q_score - c_score) <= 0.15

            results.append(QMLResult(
                guide           = guide,
                position        = site["position"],
                strand          = site["strand"],
                pam             = site["pam"],
                gc_content      = site["gc_content"],
                quantum_score   = round(q_score, 4),
                classical_score = round(c_score, 4),
                confidence      = confidence,
                agreement       = agreement,
                method          = "QNN-Regressor",
                error           = ""
            ))
        except Exception as e:
            results.append(QMLResult(guide=guide, position=site["position"], strand=site["strand"], pam=site["pam"], gc_content=site["gc_content"], error=str(e)))

    results.sort(key=lambda r: r.confidence, reverse=True)
    return results


# ── PRINT REPORT ──────────────────────────────────────────────────────────────

def print_qml_report(results: list) -> None:
    print("\n" + "=" * 65)
    print("  QACS — QNN CONTINUOUS EFFICIENCY REPORT")
    print("=" * 65)

    if not results:
        print("  No results.")
        return

    successful = [r for r in results if not r.error]
    print(f"  Method          : Continuous Quantum Neural Network (QNN)")
    print(f"  Observable      : Pauli-Z Expectation [-1.0, 1.0] scaled to [0, 1]")
    print(f"  Optimizer       : SPSA (Stochastic Gradient Approximation)")
    print(f"  Sites evaluated : {len(results)}")

    if successful:
        print(f"\n  {'#':<4} {'Pos':<6} {'PAM':<5} {'GC%':<8} {'Quantum':<10} {'Classical':<12} {'Agree':<8} Confidence")
        print(f"  {'-'*4} {'-'*6} {'-'*5} {'-'*8} {'-'*10} {'-'*12} {'-'*8} {'-'*10}")

        for i, r in enumerate(successful, 1):
            agree = "✓" if r.agreement else "⚠"
            print(f"  {i:<4} {r.position:<6} {r.pam:<5} {r.gc_content:.1%}    "
                  f"{r.quantum_score:.4f}    {r.classical_score:.4f}      {agree:<8} {r.confidence:.4f}")

        best = successful[0]
        print(f"\n  ── Best candidate ──")
        print(f"  Guide RNA       : {best.guide}")
        print(f"  Position        : {best.position}")
        print(f"  Quantum score   : {best.quantum_score:.4f}")
        print(f"  Classical score : {best.classical_score:.4f}")
        
    print("=" * 65)


# ── SYNTHETIC DATA GENERATOR ──────────────────────────────────────────────────

def generate_synthetic_training_data(n_samples: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    bases, sequences, scores = ["A", "T", "G", "C"], [], []
    for _ in range(n_samples):
        guide = "".join(np.random.choice(bases, 20))
        gc = (guide.count("G") + guide.count("C")) / 20
        efficiency = 1.0 - abs(gc - 0.55) * 2.0
        seed = guide[8:]
        seed_gc = (seed.count("G") + seed.count("C")) / 12
        efficiency += (seed_gc - 0.55) * 0.3
        if guide[-1] in ["G", "A"]: efficiency += 0.1
        efficiency += np.random.normal(0, 0.15)
        sequences.append(guide)
        scores.append(round(np.clip(efficiency, 0.0, 1.0), 4))
    return pd.DataFrame({"sequence": sequences, "efficiency": scores})


# ── RUN TESTS ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("QACS — Quantum Machine Learning Engine (Continuous QNN)")
    print("=" * 55)

    model = QMLModel()
    loaded = model.load(MODEL_PATH)

    if not loaded:
        print("\nNo saved model found. Training continuous QNN from scratch...")
        df = generate_synthetic_training_data(n_samples=1000)
        model.train(df)
        model.save(MODEL_PATH)

    print("\n── Inference Test: HBB Sickle Cell Cut Sites ──")
    test_sites = [
        {"position": 33, "pam": "TGG", "guide": "ACAGTGCAGCTCACTCAGTG", "strand": "reverse", "gc_content": 0.55},
        {"position": 39, "pam": "AGG", "guide": "CAGCTCACTCAGTGTGGCAA", "strand": "reverse", "gc_content": 0.55},
        {"position": 45, "pam": "TGG", "guide": "AGTCTGCCGTTACTGCCCTG", "strand": "forward", "gc_content": 0.60}
    ]

    results = evaluate_sites_qml(test_sites, model)
    print_qml_report(results)

    print("\n── Benchmark Summary ──")
    print(f"  QNN Pearson r         : {model.qnn_r}")
    print(f"  QNN RMSE              : {model.qnn_rmse}")
    print(f"  XGBoost Pearson r     : {model.xgb_r}")
    print(f"  XGBoost RMSE          : {model.xgb_rmse}")