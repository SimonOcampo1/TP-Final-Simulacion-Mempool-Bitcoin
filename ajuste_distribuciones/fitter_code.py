"""
FRAMEWORK ROBUSTO DE AJUSTE DE DISTRIBUCIONES CON SEGMENTACIÓN
Universidad Tecnológica Nacional - FRLP
Autor: Simón Tadeo Ocampo
Año: 2025

Framework completo para ajuste de distribuciones probabilísticas
con validación cruzada, detección de overfitting y segmentación
automática mediante búsqueda de cutoff óptimo.

Características principales:
- K-fold cross-validation para validación de ajuste
- Bootstrap para estabilidad paramétrica
- Criterios de información múltiples (AIC, BIC, HQIC)
- Detección automática de overfitting
- Segmentación jerárquica con ΔBIC
- Búsqueda de cutoff por rejilla refinada
- Métricas de error en colas

Uso:
    # Demo con datos sintéticos
    python fitter_code.py --demo
    
    # Ajuste simple de variable
    python fitter_code.py --csv datos.csv --column size_bytes
    
    # Ajuste con segmentación automática
    python fitter_code.py --csv datos.csv --column size_bytes --segment --cutoffs 25
"""

import argparse
import warnings
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


@dataclass
class FitResult:
    """Resultado del ajuste de una distribución individual"""
    distribution: str
    params: Optional[Tuple[float, ...]]
    n_params: int
    log_likelihood: float
    aic: float
    bic: float
    hqic: float
    ks_stat: float
    p_value: float
    cv_ks_mean: float
    cv_ks_std: float
    param_stability: float
    overfitting_score: int
    overfitting_flags: Dict[str, bool]
    fit_successful: bool
    fallback_applied: bool = False
    adjusted_bic: float = np.inf
    preference_bonus: float = 0.0
    quality_penalty: float = 0.0
    overfitting_penalty: float = 0.0


@dataclass
class VariableFit:
    """Resultado del ajuste de una variable con todas las distribuciones candidatas"""
    variable_name: str
    characteristics: Dict[str, float]
    best_distribution: FitResult
    all_results: List[FitResult]
    fit_quality: str


@dataclass
class SegmentFit:
    """Ajuste de un segmento individual"""
    data: pd.Series
    fit: Optional[VariableFit]


@dataclass
class SegmentationResult:
    """Resultado completo de segmentación por cutoff"""
    variable_name: str
    cutoff: float
    weights: Dict[str, float]
    segments: Dict[str, SegmentFit]
    mixture_metrics: Optional[Dict[str, float]]
    acceptance: Dict[str, bool]
    acceptance_summary: str


class RobustDistributionFitter:
    """
    Fitter robusto con soporte para segmentación y búsqueda óptima de cutoff.
    
    Metodología:
    1. Análisis de características de datos
    2. Selección de distribuciones candidatas según tipo de variable
    3. Ajuste MLE con validación cruzada K-fold
    4. Bootstrap para estabilidad paramétrica
    5. Selección mediante BIC ajustado con bonificaciones teóricas
    6. Detección de overfitting mediante umbrales conservadores
    7. Segmentación opcional con búsqueda de cutoff óptimo
    """

    _DEFAULT_SEGMENT_RULES = {
        "min_segment_size": 30,
        "delta_bic_threshold": 10.0,
        "max_param_cv": 0.30,
        "max_ks": 0.30,
    }

    def __init__(
        self,
        distributions_for_times: Optional[Sequence[str]] = None,
        distributions_for_values: Optional[Sequence[str]] = None,
        segmentation_rules: Optional[Dict[str, float]] = None,
        random_state: Optional[int] = 42,
    ) -> None:
        """
        Args:
            distributions_for_times: Distribuciones candidatas para variables temporales
            distributions_for_values: Distribuciones candidatas para variables de valor
            segmentation_rules: Reglas para aceptación de segmentación
            random_state: Semilla para reproducibilidad
        """
        self.distributions_for_times = list(
            distributions_for_times
            or ("expon", "gamma", "weibull_min", "lognorm", "invgamma", "rayleigh")
        )
        self.distributions_for_values = list(
            distributions_for_values
            or (
                "lognorm",
                "gamma",
                "weibull_min",
                "pareto",
                "burr",
                "chi2",
                "beta",
                "genpareto",
                "johnsonsu",
                "t",
            )
        )
        self.segmentation_rules = {**self._DEFAULT_SEGMENT_RULES, **(segmentation_rules or {})}
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

        self.ks_thresholds = {
            "excellent": 0.05,
            "very_good": 0.10,
            "good": 0.20,
            "acceptable": 0.30,
        }
        self.overfitting_thresholds = {
            "bic_suspicious": -8000,
            "ks_too_low": 0.02,
            "param_instability": 0.30,
        }

    @staticmethod
    def _clean_data(data: pd.Series) -> pd.Series:
        """Limpia datos removiendo valores no-positivos y NaN"""
        return data[data > 0].dropna()

    def analyze_data_characteristics(self, data: pd.Series, variable_name: str) -> Dict[str, float]:
        """
        Analiza características estadísticas de los datos para selección de distribuciones.
        
        Returns:
            Dict con sample_size, skewness, kurtosis, heavy_tails, tail_ratio, is_time_series
        """
        clean = self._clean_data(data)
        if clean.empty:
            return {
                "sample_size": 0,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "high_skewness": False,
                "high_kurtosis": False,
                "heavy_tails": False,
                "many_outliers": False,
                "tail_ratio": np.nan,
                "is_time_series": "tiempo" in variable_name.lower() or "arrival" in variable_name.lower(),
            }

        skewness = stats.skew(clean)
        kurtosis = stats.kurtosis(clean)
        q1, q3 = clean.quantile([0.25, 0.75])
        iqr = q3 - q1
        extreme_outliers = ((clean > q3 + 3 * iqr) | (clean < q1 - 3 * iqr)).sum()
        percentiles = clean.quantile([0.90, 0.95, 0.99])
        tail_ratio = float(percentiles.loc[0.99] / percentiles.loc[0.95]) if percentiles.loc[0.95] else np.nan

        return {
            "sample_size": int(len(clean)),
            "skewness": float(skewness),
            "kurtosis": float(kurtosis),
            "high_skewness": bool(abs(skewness) > 2),
            "high_kurtosis": bool(kurtosis > 7),
            "heavy_tails": bool(np.isfinite(tail_ratio) and tail_ratio > 2.5),
            "many_outliers": bool(extreme_outliers > len(clean) * 0.01),
            "tail_ratio": float(tail_ratio) if np.isfinite(tail_ratio) else np.nan,
            "is_time_series": "tiempo" in variable_name.lower() or "arrival" in variable_name.lower(),
        }

    def cross_validation_ks(
        self,
        data: pd.Series,
        distribution_name: str,
        n_splits: int = 3,
        max_samples: int = 5000,
    ) -> Dict[str, float]:
        """
        Validación cruzada K-fold calculando estadístico K-S en fold de validación.
        
        Returns:
            Dict con cv_ks_mean, cv_ks_std, cv_successful
        """
        clean = self._clean_data(data)
        if len(clean) > max_samples:
            clean = clean.sample(n=max_samples, random_state=self.random_state)
        if len(clean) < 30:
            n_splits = max(2, len(clean) // 10) if len(clean) // 10 > 1 else 2
        if len(clean) < 5 or n_splits < 2:
            return {"cv_ks_mean": np.inf, "cv_ks_std": np.inf, "cv_successful": False}

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        dist = getattr(stats, distribution_name)
        try:
            for train_idx, val_idx in kf.split(clean):
                train = clean.iloc[train_idx]
                val = clean.iloc[val_idx]
                params = dist.fit(train)
                ks_stat, _ = stats.kstest(val, lambda x: dist.cdf(x, *params))
                scores.append(ks_stat)
        except Exception:
            return {"cv_ks_mean": np.inf, "cv_ks_std": np.inf, "cv_successful": False}

        return {
            "cv_ks_mean": float(np.mean(scores)),
            "cv_ks_std": float(np.std(scores)),
            "cv_successful": True,
            "n_splits_used": n_splits,
        }

    def bootstrap_parameter_stability(
        self,
        data: pd.Series,
        distribution_name: str,
        n_bootstrap: int = 10,
        max_samples: int = 1000,
    ) -> float:
        """
        Calcula estabilidad de parámetros mediante bootstrap.
        Retorna CV promedio (std/mean) de todos los parámetros.
        """
        clean = self._clean_data(data)
        if len(clean) < 10:
            return np.inf

        dist = getattr(stats, distribution_name)
        sample_size = min(len(clean), max_samples)
        params_list = []
        for _ in range(n_bootstrap):
            if len(clean) > sample_size:
                base_sample = clean.sample(n=sample_size, random_state=self.random_state)
                sample = base_sample.sample(n=sample_size, replace=True, random_state=self.random_state)
            else:
                sample = clean.sample(n=len(clean), replace=True, random_state=self.random_state)
            try:
                params_list.append(dist.fit(sample))
            except Exception:
                return np.inf
        if len(params_list) < 3:
            return np.inf
        param_matrix = np.array(params_list)
        cv_scores = []
        for i in range(param_matrix.shape[1]):
            mean = np.mean(param_matrix[:, i])
            std = np.std(param_matrix[:, i])
            cv_scores.append(std / (abs(mean) + 1e-10))
        return float(np.mean(cv_scores))

    def fit_single_distribution(self, data: pd.Series, distribution_name: str) -> FitResult:
        """
        Ajusta una distribución específica a los datos con validación completa.
        
        Incluye:
        - Ajuste MLE
        - Cálculo de criterios de información (AIC, BIC, HQIC)
        - Test Kolmogorov-Smirnov
        - Validación cruzada K-fold
        - Bootstrap para estabilidad paramétrica
        - Detección de overfitting
        """
        clean = self._clean_data(data)
        if len(clean) < 5:
            return FitResult(
                distribution=distribution_name,
                params=None,
                n_params=0,
                log_likelihood=-np.inf,
                aic=np.inf,
                bic=np.inf,
                hqic=np.inf,
                ks_stat=np.inf,
                p_value=0.0,
                cv_ks_mean=np.inf,
                cv_ks_std=np.inf,
                param_stability=np.inf,
                overfitting_score=10,
                overfitting_flags={"insufficient_data": True},
                fit_successful=False,
            )
        dist = getattr(stats, distribution_name)
        try:
            params = dist.fit(clean)
            n_params = len(params)
            n_obs = len(clean)
            log_likelihood = float(np.sum(dist.logpdf(clean, *params)))
            aic = -2 * log_likelihood + 2 * n_params
            bic = -2 * log_likelihood + n_params * np.log(n_obs)
            hqic = -2 * log_likelihood + 2 * n_params * np.log(np.log(n_obs))
            ks_stat, p_value = stats.kstest(clean, lambda x: dist.cdf(x, *params))
            cv_metrics = self.cross_validation_ks(clean, distribution_name)
            param_stability = self.bootstrap_parameter_stability(clean, distribution_name)
            
            overfitting_flags = {
                "suspicious_bic": bic < self.overfitting_thresholds["bic_suspicious"],
                "ks_too_low": cv_metrics["cv_ks_mean"] < self.overfitting_thresholds["ks_too_low"],
                "unstable_params": param_stability > self.overfitting_thresholds["param_instability"],
                "impossible_likelihood": log_likelihood > 0,
            }
            overfitting_score = int(sum(overfitting_flags.values()))
            
            return FitResult(
                distribution=distribution_name,
                params=tuple(float(p) for p in params),
                n_params=n_params,
                log_likelihood=log_likelihood,
                aic=float(aic),
                bic=float(bic),
                hqic=float(hqic),
                ks_stat=float(ks_stat),
                p_value=float(p_value),
                cv_ks_mean=float(cv_metrics["cv_ks_mean"]),
                cv_ks_std=float(cv_metrics["cv_ks_std"]),
                param_stability=float(param_stability),
                overfitting_score=overfitting_score,
                overfitting_flags=overfitting_flags,
                fit_successful=True,
            )
        except Exception as exc:
            return FitResult(
                distribution=distribution_name,
                params=None,
                n_params=0,
                log_likelihood=-np.inf,
                aic=np.inf,
                bic=np.inf,
                hqic=np.inf,
                ks_stat=np.inf,
                p_value=0.0,
                cv_ks_mean=np.inf,
                cv_ks_std=np.inf,
                param_stability=np.inf,
                overfitting_score=10,
                overfitting_flags={"fit_error": True, "message": str(exc)},
                fit_successful=False,
            )

    def intelligent_selection(
        self,
        results: Iterable[FitResult],
        characteristics: Dict[str, float],
    ) -> Tuple[FitResult, List[FitResult]]:
        """
        Selección inteligente de distribución mediante BIC ajustado.
        
        Aplica bonificaciones teóricas según tipo de variable y características.
        Penaliza por pobre validación cruzada y overfitting detectado.
        Aplica fallback si la mejor distribución presenta overfitting severo.
        """
        valid = [r for r in results if r.fit_successful]
        if not valid:
            raise ValueError("Ninguna distribución se ajustó exitosamente")

        bonuses = self._build_preference_bonuses(characteristics)
        scored: List[FitResult] = []
        for result in valid:
            bonus = bonuses.get(result.distribution, 0.0)
            if result.cv_ks_mean > 0.4:
                penalty = 2000.0
            elif result.cv_ks_mean > 0.25:
                penalty = 500.0
            else:
                penalty = 0.0
            overfitting_penalty = result.overfitting_score * 300.0
            adjusted_bic = result.bic + bonus + penalty + overfitting_penalty
            scored.append(
                replace(
                    result,
                    adjusted_bic=adjusted_bic,
                    preference_bonus=bonus,
                    quality_penalty=penalty,
                    overfitting_penalty=overfitting_penalty,
                )
            )

        scored.sort(key=lambda r: r.__dict__["adjusted_bic"])
        best = scored[0]
        
        if best.overfitting_score >= 3 and len(scored) > 1:
            fallback = scored[1]
            fallback.fallback_applied = True
            return fallback, scored
        
        best.fallback_applied = False
        return best, scored

    def _build_preference_bonuses(self, characteristics: Dict[str, float]) -> Dict[str, float]:
        """
        Bonificaciones teóricas según tipo de variable y características.
        Valores negativos = bonificación (menor BIC ajustado).
        Valores positivos = penalización (mayor BIC ajustado).
        """
        if characteristics.get("is_time_series", False):
            return {
                "expon": -100 if not characteristics.get("high_skewness") else 200,
                "gamma": -300,
                "weibull_min": -200,
                "lognorm": -400 if characteristics.get("high_skewness") else 100,
                "invgamma": -250,
                "rayleigh": -150,
            }
        return {
            "lognorm": -500,
            "gamma": -200,
            "weibull_min": -100,
            "pareto": -700 if characteristics.get("heavy_tails") else 300,
            "burr": -500 if characteristics.get("high_kurtosis") else 150,
            "chi2": -50,
            "beta": 150,
            "genpareto": -650 if characteristics.get("heavy_tails") else 250,
            "johnsonsu": -600 if (characteristics.get("high_skewness") or characteristics.get("high_kurtosis")) else 200,
            "t": -400 if characteristics.get("heavy_tails") else 100,
        }

    def classify_fit_quality(self, cv_ks_mean: float) -> str:
        """Clasifica calidad del ajuste según K-S de validación cruzada"""
        if cv_ks_mean < self.ks_thresholds["excellent"]:
            return "Excelente"
        if cv_ks_mean < self.ks_thresholds["very_good"]:
            return "Muy bueno"
        if cv_ks_mean < self.ks_thresholds["good"]:
            return "Bueno"
        if cv_ks_mean < self.ks_thresholds["acceptable"]:
            return "Aceptable"
        return "Inaceptable"

    def fit_variable(self, data: pd.Series, variable_name: str) -> VariableFit:
        """
        Ajuste completo de una variable con selección automática de mejor distribución.
        
        Returns:
            VariableFit con mejor distribución y todos los resultados ordenados por BIC ajustado
        """
        characteristics = self.analyze_data_characteristics(data, variable_name)
        candidates = (
            self.distributions_for_times if characteristics["is_time_series"] else self.distributions_for_values
        )
        results = [self.fit_single_distribution(data, dist) for dist in candidates]
        best, scored = self.intelligent_selection(results, characteristics)
        return VariableFit(
            variable_name=variable_name,
            characteristics=characteristics,
            best_distribution=best,
            all_results=scored,
            fit_quality=self.classify_fit_quality(best.cv_ks_mean),
        )

    def fit_segmented_variable(
        self,
        data: pd.Series,
        variable_name: str,
        cutoff_point: float,
        min_segment_size: Optional[int] = None,
        n_bootstrap_segment: int = 8,
        reference_fit: Optional[VariableFit] = None,
    ) -> Optional[SegmentationResult]:
        """
        Ajusta variable con segmentación por cutoff fijo.
        
        Evalúa aceptación mediante criterios jerárquicos:
        1. ΔBIC negativo significativo
        2. Estabilidad paramétrica en cada segmento
        3. Bondad de ajuste (K-S) en cada segmento
        
        Returns:
            SegmentationResult con ajustes de ambos segmentos y métricas de mezcla
        """
        min_segment_size = min_segment_size or self.segmentation_rules["min_segment_size"]
        clean = self._clean_data(data)
        if clean.empty:
            return None
        seg_small = clean[clean < cutoff_point]
        seg_large = clean[clean >= cutoff_point]
        if len(seg_small) < min_segment_size or len(seg_large) < min_segment_size:
            return None

        result_small = self.fit_variable(seg_small, f"{variable_name} - Pequeño")
        result_large = self.fit_variable(seg_large, f"{variable_name} - Grande")

        for res, seg_data in ((result_small, seg_small), (result_large, seg_large)):
            if res is None or not res.best_distribution.fit_successful or res.best_distribution.params is None:
                continue
            stability = self.bootstrap_parameter_stability(
                seg_data,
                res.best_distribution.distribution,
                n_bootstrap=n_bootstrap_segment,
            )
            res.best_distribution.param_stability = float(stability)

        weights = {
            "small": len(seg_small) / len(clean),
            "large": len(seg_large) / len(clean),
        }
        mixture_metrics = self._compute_mixture_metrics(result_small, result_large, weights, seg_small, seg_large)
        if reference_fit is None:
            reference_fit = self.fit_variable(clean, variable_name)
        acceptance = self._evaluate_segmentation_acceptance(
            result_small,
            result_large,
            mixture_metrics,
            reference_bic=reference_fit.best_distribution.bic,
        )
        summary = self._summarize_acceptance(acceptance)
        return SegmentationResult(
            variable_name=variable_name,
            cutoff=float(cutoff_point),
            weights=weights,
            segments={
                "small": SegmentFit(data=seg_small, fit=result_small),
                "large": SegmentFit(data=seg_large, fit=result_large),
            },
            mixture_metrics=mixture_metrics,
            acceptance=acceptance,
            acceptance_summary=summary,
        )

    def _compute_mixture_metrics(
        self,
        fit_small: VariableFit,
        fit_large: VariableFit,
        weights: Dict[str, float],
        data_small: pd.Series,
        data_large: pd.Series,
    ) -> Optional[Dict[str, float]]:
        """Calcula log-likelihood y criterios de información para modelo mezcla"""
        if not fit_small or not fit_large:
            return None
        best_small = fit_small.best_distribution
        best_large = fit_large.best_distribution
        if not best_small.params or not best_large.params:
            return None
        dist_small = getattr(stats, best_small.distribution)
        dist_large = getattr(stats, best_large.distribution)
        data_all = pd.concat([data_small, data_large]).sort_values()
        pdf_small = dist_small.pdf(data_all, *best_small.params)
        pdf_large = dist_large.pdf(data_all, *best_large.params)
        mixture_pdf = weights["small"] * pdf_small + weights["large"] * pdf_large + 1e-30
        log_likelihood = float(np.sum(np.log(mixture_pdf)))
        n_params_mix = best_small.n_params + best_large.n_params + 1
        n_obs = len(data_all)
        aic = -2 * log_likelihood + 2 * n_params_mix
        bic = -2 * log_likelihood + n_params_mix * np.log(n_obs)
        hqic = -2 * log_likelihood + 2 * n_params_mix * np.log(np.log(n_obs))
        return {
            "log_likelihood": log_likelihood,
            "aic": float(aic),
            "bic": float(bic),
            "hqic": float(hqic),
            "n_params": int(n_params_mix),
            "n_obs": int(n_obs),
        }

    def _evaluate_segmentation_acceptance(
        self,
        fit_small: VariableFit,
        fit_large: VariableFit,
        mixture_metrics: Optional[Dict[str, float]],
        reference_bic: float,
    ) -> Dict[str, bool]:
        """
        Evalúa criterios jerárquicos de aceptación de segmentación.
        
        Criterios:
        1. ΔBIC < -threshold (mejora significativa sobre modelo único)
        2. CV parámetros < max_param_cv (estabilidad en cada segmento)
        3. K-S < max_ks (bondad de ajuste en cada segmento)
        """
        rules = self.segmentation_rules
        acceptance = {
            "valid_segments": bool(fit_small and fit_large and mixture_metrics),
            "delta_bic": False,
            "cv_small": False,
            "cv_large": False,
            "ks_small": False,
            "ks_large": False,
        }
        if not acceptance["valid_segments"]:
            return acceptance
        delta_bic = mixture_metrics["bic"] - reference_bic
        acceptance["delta_bic"] = delta_bic < -abs(rules["delta_bic_threshold"])
        for label, fit in ("small", fit_small), ("large", fit_large):
            best = fit.best_distribution
            acceptance[f"cv_{label}"] = best.param_stability < rules["max_param_cv"]
            acceptance[f"ks_{label}"] = best.cv_ks_mean < rules["max_ks"]
        return acceptance

    @staticmethod
    def _summarize_acceptance(flags: Dict[str, bool]) -> str:
        """Resume resultado de evaluación jerárquica"""
        if not flags.get("valid_segments", False):
            return "Segmentación inválida (segmentos o mezcla no disponibles)."
        checks = [k for k in flags if k != "valid_segments"]
        if all(flags[k] for k in checks):
            return "✅ Segmentación aceptada: cumple todos los criterios jerárquicos."
        failed = [k for k in checks if not flags[k]]
        return "❌ Segmentación rechazada: falla en " + ", ".join(failed)

    def optimize_cutoff(
        self,
        data: pd.Series,
        variable_name: str,
        candidate_cutoffs: Optional[Sequence[float]] = None,
        n_quantiles: int = 20,
        min_segment_size: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Búsqueda de cutoff óptimo mediante evaluación de rejilla de candidatos.
        
        Args:
            data: Serie con datos a segmentar
            variable_name: Nombre de la variable
            candidate_cutoffs: Lista explícita de cutoffs (opcional)
            n_quantiles: Número de cuantiles para rejilla (si candidate_cutoffs=None)
            min_segment_size: Tamaño mínimo de segmento
            
        Returns:
            Dict con 'best' (mejor evaluación), 'evaluations' (todos los cutoffs),
            'single_fit' (ajuste sin segmentar)
        """
        clean = self._clean_data(data)
        min_segment_size = min_segment_size or self.segmentation_rules["min_segment_size"]
        if len(clean) < min_segment_size * 2:
            return {"best": None, "evaluations": []}

        if candidate_cutoffs is None:
            candidate_cutoffs = self._build_cutoff_grid(clean, n_quantiles, min_segment_size)
        evaluations = []
        best_eval = None
        best_bic = np.inf
        single_fit = self.fit_variable(clean, variable_name)
        ref_bic = single_fit.best_distribution.bic

        for cutoff in candidate_cutoffs:
            seg_result = self.fit_segmented_variable(
                clean,
                variable_name,
                cutoff,
                min_segment_size=min_segment_size,
                reference_fit=single_fit,
            )
            if not seg_result or not seg_result.mixture_metrics:
                evaluations.append({"cutoff": cutoff, "valid": False})
                continue
            bic_mix = seg_result.mixture_metrics["bic"]
            delta_bic = bic_mix - ref_bic
            evaluation = {
                "cutoff": float(cutoff),
                "valid": True,
                "bic_mix": bic_mix,
                "delta_bic": delta_bic,
                "segmentation": seg_result,
            }
            evaluations.append(evaluation)
            if np.isfinite(bic_mix) and bic_mix < best_bic:
                best_bic = bic_mix
                best_eval = evaluation

        return {
            "best": best_eval,
            "evaluations": evaluations,
            "single_fit": single_fit,
        }

    def _build_cutoff_grid(
        self,
        clean_data: pd.Series,
        n_quantiles: int,
        min_segment_size: int,
    ) -> List[float]:
        """
        Construye rejilla refinada de candidatos de cutoff.
        
        1. Genera rejilla base en cuantiles 55%-95%
        2. Filtra por tamaño mínimo de segmento
        3. Refina agregando puntos medios entre candidatos consecutivos
        """
        quantile_grid = np.linspace(0.55, 0.95, n_quantiles)
        candidates = []
        for q in quantile_grid:
            value = float(clean_data.quantile(q))
            if not np.isfinite(value):
                continue
            lower_count = int((clean_data < value).sum())
            upper_count = int((clean_data >= value).sum())
            if lower_count >= min_segment_size and upper_count >= min_segment_size:
                candidates.append(value)
        unique_candidates = sorted(set(candidates))
        if not unique_candidates:
            return [float(clean_data.median())]
        
        refined = set(unique_candidates)
        for a, b in zip(unique_candidates[:-1], unique_candidates[1:]):
            mid = 0.5 * (a + b)
            if mid not in refined:
                lower_count = int((clean_data < mid).sum())
                upper_count = int((clean_data >= mid).sum())
                if lower_count >= min_segment_size and upper_count >= min_segment_size:
                    refined.add(mid)
        return sorted(refined)

    def tail_error_metrics(
        self,
        data: pd.Series,
        single_fit: FitResult,
        segmented_results: SegmentationResult,
        quantiles: Sequence[float] = (0.9, 0.95, 0.99, 0.995),
    ) -> Dict[str, Dict[float, float]]:
        """
        Calcula métricas de error en colas comparando modelo único vs mezcla.
        
        Retorna cuantiles empíricos, teóricos (single), teóricos (mixture)
        y errores relativos para cada cuantil.
        """
        clean = self._clean_data(data)
        empirical = {q: float(np.quantile(clean, q)) for q in quantiles}
        dist_single = getattr(stats, single_fit.distribution)
        q_single = {q: float(dist_single.ppf(q, *single_fit.params)) for q in quantiles}
        if not segmented_results or not segmented_results.mixture_metrics:
            q_mix = {q: np.nan for q in quantiles}
        else:
            q_mix = self._invert_mixture_quantiles(segmented_results, quantiles, clean)
        err_single = {q: abs(q_single[q] - empirical[q]) / (empirical[q] + 1e-12) for q in quantiles}
        err_mix = {
            q: abs(q_mix[q] - empirical[q]) / (empirical[q] + 1e-12) if np.isfinite(q_mix[q]) else np.nan
            for q in quantiles
        }
        return {
            "empirical": empirical,
            "single": q_single,
            "mixture": q_mix,
            "err_single": err_single,
            "err_mixture": err_mix,
        }

    def _invert_mixture_quantiles(
        self,
        segmented_results: SegmentationResult,
        quantiles: Sequence[float],
        data: pd.Series,
    ) -> Dict[float, float]:
        """Inversión numérica de CDF de mezcla para calcular cuantiles teóricos"""
        weights = segmented_results.weights
        seg_info = segmented_results.segments
        sb = seg_info["small"].fit.best_distribution
        lb = seg_info["large"].fit.best_distribution
        dist_small = getattr(stats, sb.distribution)
        dist_large = getattr(stats, lb.distribution)

        def F_mix(x: float) -> float:
            return weights["small"] * dist_small.cdf(x, *sb.params) + weights["large"] * dist_large.cdf(x, *lb.params)

        lo = float(data.min())
        hi = float(data.quantile(0.999))
        quantile_values: Dict[float, float] = {}
        for q in quantiles:
            a, b = lo, hi
            for _ in range(60):
                mid = 0.5 * (a + b)
                if F_mix(mid) < q:
                    a = mid
                else:
                    b = mid
            quantile_values[q] = float(0.5 * (a + b))
        return quantile_values

    def generate_report(
        self,
        output_path: str,
        source_csv: str,
        variable_results: Sequence[VariableFit],
        size_single: Optional[VariableFit] = None,
        segmentation_size: Optional[SegmentationResult] = None,
        tail_metrics: Optional[Dict[str, Dict[float, float]]] = None,
    ) -> None:
        """Genera reporte de texto completo con todos los resultados"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("=" * 90 + "\n")
            f.write("REPORTE ROBUSTO UNIFICADO - MEMPOOL BITCOIN\n")
            f.write("=" * 90 + "\n")
            f.write(f"Archivo origen: {source_csv}\nFecha: {timestamp}\n\n")
            f.write(self._render_methodology_block())
            f.write(self._render_variable_summary(variable_results))
            if size_single:
                f.write(self._render_single_size_block(size_single))
            if segmentation_size:
                f.write(self._render_segmentation_block(segmentation_size, size_single))
            if tail_metrics:
                f.write(self._render_tail_metrics_block(tail_metrics))
            f.write("=" * 90 + "\nFIN DEL REPORTE COMPLETO\n" + "=" * 90 + "\n")

    def _render_methodology_block(self) -> str:
        return (
            "METODOLOGÍA COMPLETA:\n"
            "- Validación cruzada 3-fold con K-S (acelerada para datasets grandes)\n"
            "- Bootstrap 8-10 iteraciones para estabilidad paramétrica\n"
            "- Criterios de información: AIC, BIC, HQIC\n"
            "- Detección automática de overfitting con umbrales conservadores\n"
            "- Selección con bonificaciones teóricas según tipo de variable\n"
            "- Segmentación jerárquica con ΔBIC, CV parámetros y K-S\n"
            "- Búsqueda de cutoff por rejilla refinada en cuantiles 55%-95%\n\n"
        )

    def _render_variable_summary(self, variable_results: Sequence[VariableFit]) -> str:
        lines = ["=" * 60 + "\nRESUMEN EJECUTIVO\n" + "=" * 60 + "\n"]
        for res in variable_results:
            best = res.best_distribution
            fallback = " ⚠️ FALLBACK" if best.fallback_applied else ""
            lines.append(f"🔍 {res.variable_name}:\n")
            lines.append(f"  Distribución: {best.distribution.upper()}{fallback}\n")
            lines.append(f"  Calidad: {res.fit_quality} (KS={best.cv_ks_mean:.4f})\n")
            lines.append(f"  BIC={best.bic:.2f} | AIC={best.aic:.2f}\n")
            lines.append(f"  Estabilidad parámetros (CV): {best.param_stability:.4f}\n")
            if best.params:
                formatted_params = ", ".join(f"{p:.6g}" for p in best.params)
                lines.append(f"  Parámetros: {formatted_params}\n")
            lines.append(f"  Tamaño muestra: {res.characteristics['sample_size']:,}\n\n")
        return "".join(lines)

    def _render_single_size_block(self, size_result: VariableFit) -> str:
        best = size_result.best_distribution
        lines = ["=" * 60 + "\nANÁLISIS DETALLADO: size_bytes (MODELO ÚNICO)\n" + "=" * 60 + "\n"]
        lines.append(f"📊 Distribución: {best.distribution.upper()}\n")
        lines.append(f"📈 KS (cv): {best.cv_ks_mean:.4f} ± {best.cv_ks_std:.4f}\n")
        lines.append(f"📈 P-valor K-S: {best.p_value:.6f}\n")
        lines.append(f"📊 BIC={best.bic:.2f} | AIC={best.aic:.2f} | HQIC={best.hqic:.2f}\n")
        lines.append(f"⚖️  Estabilidad bootstrap: {best.param_stability:.4f}\n")
        if best.params:
            formatted_params = ", ".join(f"{p:.8g}" for p in best.params)
            lines.append(f"🔢 Parámetros: {formatted_params}\n")
        chars = size_result.characteristics
        lines.append(f"📈 Asimetría: {chars['skewness']:.4f}, Curtosis: {chars['kurtosis']:.4f}\n")
        lines.append(f"📊 Tamaño muestra: {chars['sample_size']:,}\n\n")
        return "".join(lines)

    def _render_segmentation_block(
        self,
        segmentation: SegmentationResult,
        single_result: Optional[VariableFit],
    ) -> str:
        lines = ["=" * 60 + "\nANÁLISIS SEGMENTADO: size_bytes\n" + "=" * 60 + "\n"]
        lines.append(f"🔪 Cutoff: {segmentation.cutoff:.2f}\n")
        lines.append(
            f"   Peso segmento pequeño: {segmentation.weights['small']:.2%} | grande: {segmentation.weights['large']:.2%}\n"
        )
        for key, label, emoji in (("small", "Pequeño", "🔸"), ("large", "Grande", "🔹")):
            seg = segmentation.segments[key]
            if not seg.fit:
                lines.append(f"{emoji} Segmento {label}: ❌ sin ajuste válido\n\n")
                continue
            best = seg.fit.best_distribution
            lines.append(f"{emoji} Segmento {label}: {best.distribution.upper()}\n")
            lines.append(
                f"   KS={best.cv_ks_mean:.4f} ± {best.cv_ks_std:.4f} | CV parámetros={best.param_stability:.4f}\n"
            )
            lines.append(f"   BIC={best.bic:.2f} | AIC={best.aic:.2f}\n")
            if best.params:
                formatted_params = ", ".join(f"{p:.8g}" for p in best.params)
                lines.append(f"   Parámetros: {formatted_params}\n")
            lines.append(f"   Obs: {len(seg.data):,}\n\n")
        if segmentation.mixture_metrics and single_result:
            mix = segmentation.mixture_metrics
            delta_bic = mix["bic"] - single_result.best_distribution.bic
            lines.append("🔀 Mezcla vs modelo único:\n")
            lines.append(f"   BIC mezcla: {mix['bic']:.2f} (ΔBIC={delta_bic:.2f})\n")
            lines.append(f"   AIC mezcla: {mix['aic']:.2f}\n")
        lines.append(f"🏁 Evaluación jerárquica: {segmentation.acceptance_summary}\n\n")
        return "".join(lines)

    def _render_tail_metrics_block(self, tail_metrics: Dict[str, Dict[float, float]]) -> str:
        lines = ["=" * 60 + "\nEVALUACIÓN DE COLAS\n" + "=" * 60 + "\n"]
        for q in tail_metrics["empirical"].keys():
            lines.append(
                f"Quantil {q:.3f}: empírico={tail_metrics['empirical'][q]:.4f} | "
                f"único={tail_metrics['single'][q]:.4f} | mezcla={tail_metrics['mixture'][q]:.4f} | "
                f"error único={tail_metrics['err_single'][q]:.2%} | "
                f"error mezcla={tail_metrics['err_mixture'][q] if np.isfinite(tail_metrics['err_mixture'][q]) else np.nan:.2%}\n"
            )
        lines.append("\n")
        return "".join(lines)

    def plot_segmented_fit(self, full_data: pd.Series, segmented_results: SegmentationResult) -> None:
        """Genera visualización del modelo segmentado con PDFs de cada segmento y mezcla"""
        clean = self._clean_data(full_data)
        x = np.linspace(clean.min(), clean.quantile(0.995), 800)
        plt.figure(figsize=(12, 7))
        plt.hist(clean, bins=120, density=True, alpha=0.55, color="#AED6F1", label="Datos")
        total_pdf = np.zeros_like(x)
        for key, label, color in (("small", "Pequeño", "#27AE60"), ("large", "Grande", "#CB4335")):
            seg = segmented_results.segments[key]
            if not seg.fit:
                continue
            best = seg.fit.best_distribution
            if not best.params:
                continue
            dist = getattr(stats, best.distribution)
            pdf_vals = dist.pdf(x, *best.params)
            weighted_pdf = pdf_vals * segmented_results.weights[key]
            total_pdf += weighted_pdf
            plt.plot(x, weighted_pdf, "-", lw=2, color=color, label=f"{best.distribution} ({label}) * peso")
        plt.plot(x, total_pdf, "k--", lw=2.2, label="Mezcla")
        plt.axvline(segmented_results.cutoff, color="black", lw=1.5, ls=":", label=f"Cutoff {segmented_results.cutoff:.2f}")
        plt.title(f"Modelo segmentado: {segmented_results.variable_name}")
        plt.xlabel("Valor")
        plt.ylabel("Densidad")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    def create_segmented_qq_plot(self, segmented_results: SegmentationResult) -> None:
        """Genera Q-Q plot separado para cada segmento"""
        plt.figure(figsize=(8, 6))
        for key, color, label in (("small", "#27AE60", "Pequeño"), ("large", "#CB4335", "Grande")):
            seg = segmented_results.segments[key]
            if not seg.fit or not seg.fit.best_distribution.params:
                continue
            clean = self._clean_data(seg.data)
            if len(clean) < 5:
                continue
            dist_name = seg.fit.best_distribution.distribution
            params = seg.fit.best_distribution.params
            dist = getattr(stats, dist_name)
            probs = (np.arange(1, len(clean) + 1) - 0.5) / len(clean)
            theo = dist.ppf(probs, *params)
            emp = np.sort(clean)
            plt.scatter(theo, emp, s=14, alpha=0.55, color=color, label=f"{label}: {dist_name}")
        all_points = np.concatenate([coll.get_offsets() for coll in plt.gca().collections if len(coll.get_offsets())])
        mn = np.nanmin(all_points)
        mx = np.nanmax(all_points)
        plt.plot([mn, mx], [mn, mx], "k--", lw=1.2, label="Identidad")
        plt.title(f"Q-Q segmentado (cutoff={segmented_results.cutoff:.2f})")
        plt.xlabel("Cuantiles teóricos")
        plt.ylabel("Cuantiles empíricos")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def _print_variable_summary(result: VariableFit) -> None:
    """Imprime resumen de ajuste de variable en consola"""
    best = result.best_distribution
    print(f"📊 Variable: {result.variable_name}")
    print(f"   Distribución seleccionada: {best.distribution.upper()}")
    print(f"   Calidad (KS): {best.cv_ks_mean:.4f} ± {best.cv_ks_std:.4f} → {result.fit_quality}")
    print(f"   BIC={best.bic:.2f} | AIC={best.aic:.2f} | HQIC={best.hqic:.2f}")
    print(f"   Estabilidad parámetros (CV): {best.param_stability:.4f}")
    if best.params:
        params = ", ".join(f"{p:.6g}" for p in best.params)
        print(f"   Parámetros: {params}")
    print()


def _print_segmentation_summary(opt_result: Dict[str, object]) -> None:
    """Imprime resumen de resultado de optimización de cutoff"""
    best_eval = opt_result.get("best")
    if not best_eval:
        print("⚠️  No se encontró un cutoff válido que cumpla los criterios de segmentación.")
        return
    seg: SegmentationResult = best_eval["segmentation"]
    print(f"🔪 Mejor cutoff: {best_eval['cutoff']:.4f}")
    print(f"   ΔBIC respecto modelo único: {best_eval['delta_bic']:.2f}")
    print(f"   Evaluación jerárquica: {seg.acceptance_summary}")
    for key in ("small", "large"):
        seg_fit = seg.segments[key].fit
        if not seg_fit:
            print(f"   Segmento {key}: sin ajuste válido")
            continue
        best = seg_fit.best_distribution
        print(
            f"   Segmento {key}: {best.distribution} | KS={best.cv_ks_mean:.4f} | CV={best.param_stability:.4f}"
        )
    print()


def _run_demo(fitter: RobustDistributionFitter) -> None:
    """Ejecuta demostración con datos sintéticos"""
    print("Ejecutando demo rápida con datos sintéticos...")
    rng = np.random.default_rng(2025)
    small = rng.lognormal(mean=5.2, sigma=0.35, size=700)
    large = rng.lognormal(mean=7.1, sigma=0.55, size=300)
    data = pd.Series(np.concatenate([small, large]), name="size_bytes")
    single_fit = fitter.fit_variable(data, data.name)
    _print_variable_summary(single_fit)
    opt_result = fitter.optimize_cutoff(data, data.name, n_quantiles=15)
    _print_segmentation_summary(opt_result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fitter robusto para distribuciones y segmentación por cutoff"
    )
    parser.add_argument("--demo", action="store_true", help="Ejecuta una demostración con datos sintéticos")
    parser.add_argument("--csv", type=Path, help="Ruta a un CSV con la columna a analizar")
    parser.add_argument("--column", help="Nombre de la columna del CSV a ajustar")
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Intenta encontrar el mejor cutoff y muestra los resultados de segmentación",
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        default=20,
        help="Cantidad de cuantiles para explorar como candidatos de cutoff (por defecto: 20)",
    )
    args = parser.parse_args()

    if not args.demo and not args.csv:
        parser.print_help()
        return

    fitter = RobustDistributionFitter()

    if args.demo:
        _run_demo(fitter)
        if not args.csv:
            return

    if args.csv:
        csv_path: Path = args.csv
        if not csv_path.exists():
            parser.error(f"No se encontró el archivo: {csv_path}")
        if not args.column:
            parser.error("Debe indicar --column con el nombre de la columna a analizar")

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            parser.error(f"No se pudo leer el CSV ({exc})")

        if args.column not in df.columns:
            parser.error(f"La columna '{args.column}' no existe en el CSV")

        series = df[args.column].astype(float)
        variable_fit = fitter.fit_variable(series, args.column)
        _print_variable_summary(variable_fit)

        if args.segment:
            opt_result = fitter.optimize_cutoff(series, args.column, n_quantiles=args.cutoffs)
            _print_segmentation_summary(opt_result)


if __name__ == "__main__":
    main()
