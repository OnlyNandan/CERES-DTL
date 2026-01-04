"""
CERES Advanced ML Engine
========================
Sophisticated agricultural calculations and ML models for:
- Evapotranspiration (Penman-Monteith)
- Growing Degree Days (GDD)
- Yield Prediction
- Disease/Pest Risk Assessment
- Soil Health Index
- Irrigation Scheduling
- Crop Suitability Analysis
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
import os

from config import Config


class EvapotranspirationCalculator:
    """
    FAO-56 Penman-Monteith Reference Evapotranspiration (ET0) Calculator
    
    Based on: Allen et al. (1998) FAO Irrigation and Drainage Paper No. 56
    """
    
    def __init__(self):
        self.stefan_boltzmann = Config.STEFAN_BOLTZMANN
        self.psychrometric = Config.PSYCHROMETRIC_CONST
        self.solar_constant = Config.SOLAR_CONSTANT
    
    def calculate_et0(
        self,
        temp_max: float,
        temp_min: float,
        humidity: float,
        wind_speed: float,
        solar_radiation: float = None,
        latitude: float = 20.0,
        elevation: float = 100.0,
        day_of_year: int = None
    ) -> Dict:
        """
        Calculate reference evapotranspiration using Penman-Monteith equation.
        
        Args:
            temp_max: Maximum daily temperature (°C)
            temp_min: Minimum daily temperature (°C)
            humidity: Relative humidity (%)
            wind_speed: Wind speed at 2m height (m/s)
            solar_radiation: Solar radiation (MJ/m²/day), estimated if None
            latitude: Latitude in degrees
            elevation: Elevation above sea level (m)
            day_of_year: Julian day (1-365)
        
        Returns:
            Dict with ET0 and intermediate calculations
        """
        if day_of_year is None:
            day_of_year = datetime.now().timetuple().tm_yday
        
        # Mean temperature
        temp_mean = (temp_max + temp_min) / 2
        
        # Atmospheric pressure (kPa)
        pressure = 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26
        
        # Psychrometric constant
        gamma = 0.000665 * pressure
        
        # Saturation vapor pressure (kPa)
        e_sat_max = 0.6108 * math.exp((17.27 * temp_max) / (temp_max + 237.3))
        e_sat_min = 0.6108 * math.exp((17.27 * temp_min) / (temp_min + 237.3))
        e_sat = (e_sat_max + e_sat_min) / 2
        
        # Actual vapor pressure
        e_actual = e_sat * (humidity / 100)
        
        # Vapor pressure deficit
        vpd = e_sat - e_actual
        
        # Slope of saturation vapor pressure curve
        delta = (4098 * e_sat) / ((temp_mean + 237.3) ** 2)
        
        # Estimate solar radiation if not provided
        if solar_radiation is None:
            solar_radiation = self._estimate_solar_radiation(
                temp_max, temp_min, latitude, day_of_year
            )
        
        # Net radiation calculation
        # Extraterrestrial radiation
        lat_rad = math.radians(latitude)
        dr = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)
        solar_decl = 0.409 * math.sin(2 * math.pi * day_of_year / 365 - 1.39)
        
        ws = math.acos(-math.tan(lat_rad) * math.tan(solar_decl))
        Ra = (24 * 60 / math.pi) * self.solar_constant * dr * (
            ws * math.sin(lat_rad) * math.sin(solar_decl) +
            math.cos(lat_rad) * math.cos(solar_decl) * math.sin(ws)
        )
        
        # Clear sky radiation
        Rso = (0.75 + 2e-5 * elevation) * Ra
        
        # Net shortwave radiation
        Rns = (1 - 0.23) * solar_radiation
        
        # Net longwave radiation
        Rnl = self.stefan_boltzmann * (
            ((temp_max + 273.16) ** 4 + (temp_min + 273.16) ** 4) / 2
        ) * (0.34 - 0.14 * math.sqrt(e_actual)) * (
            1.35 * solar_radiation / max(Rso, 0.1) - 0.35
        )
        
        # Net radiation
        Rn = Rns - Rnl
        
        # Soil heat flux (assumed 0 for daily calculations)
        G = 0
        
        # FAO-56 Penman-Monteith Equation
        et0_num = 0.408 * delta * (Rn - G) + gamma * (900 / (temp_mean + 273)) * wind_speed * vpd
        et0_den = delta + gamma * (1 + 0.34 * wind_speed)
        et0 = et0_num / et0_den
        
        return {
            'et0': round(max(et0, 0), 2),  # mm/day
            'vapor_pressure_deficit': round(vpd, 3),
            'saturation_vapor_pressure': round(e_sat, 3),
            'net_radiation': round(Rn, 2),
            'extraterrestrial_radiation': round(Ra, 2),
            'soil_heat_flux': G,
            'psychrometric_constant': round(gamma, 4)
        }
    
    def _estimate_solar_radiation(
        self,
        temp_max: float,
        temp_min: float,
        latitude: float,
        day_of_year: int
    ) -> float:
        """Hargreaves-Samani solar radiation estimation"""
        lat_rad = math.radians(latitude)
        dr = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)
        solar_decl = 0.409 * math.sin(2 * math.pi * day_of_year / 365 - 1.39)
        ws = math.acos(-math.tan(lat_rad) * math.tan(solar_decl))
        
        Ra = (24 * 60 / math.pi) * self.solar_constant * dr * (
            ws * math.sin(lat_rad) * math.sin(solar_decl) +
            math.cos(lat_rad) * math.cos(solar_decl) * math.sin(ws)
        )
        
        # Hargreaves coefficient (calibrated for Indian conditions)
        krs = 0.17  # 0.16-0.19 for interior, 0.19 for coastal
        Rs = krs * math.sqrt(temp_max - temp_min) * Ra
        
        return Rs
    
    def calculate_crop_et(
        self,
        et0: float,
        crop: str,
        growth_stage: str = 'mid'
    ) -> float:
        """Calculate crop-specific evapotranspiration (ETc)"""
        kc = Config.CROP_KC.get(crop, {}).get(growth_stage, 1.0)
        return round(et0 * kc, 2)


class GrowingDegreeDaysCalculator:
    """
    Growing Degree Days (GDD) / Heat Units Calculator
    
    Methods: Standard, Modified, Ontario
    """
    
    def __init__(self):
        self.base_temps = Config.GDD_BASE_TEMPS
    
    def calculate_gdd(
        self,
        temp_max: float,
        temp_min: float,
        crop: str,
        method: str = 'modified'
    ) -> Dict:
        """
        Calculate Growing Degree Days for a crop.
        
        Args:
            temp_max: Maximum daily temperature (°C)
            temp_min: Minimum daily temperature (°C)
            crop: Crop name
            method: 'standard', 'modified', or 'ontario'
        
        Returns:
            Dict with GDD values and growth stage info
        """
        base_temp = self.base_temps.get(crop.lower(), 10)
        upper_threshold = 30  # Most crops stop growing above this
        
        if method == 'standard':
            gdd = max(0, ((temp_max + temp_min) / 2) - base_temp)
        
        elif method == 'modified':
            # Clip temperatures
            t_max = min(temp_max, upper_threshold)
            t_min = max(temp_min, base_temp)
            t_max = max(t_max, t_min)
            gdd = (t_max + t_min) / 2 - base_temp
            gdd = max(0, gdd)
        
        elif method == 'ontario':
            # More sophisticated method for corn/maize
            t_max = min(max(temp_max, base_temp), upper_threshold)
            t_min = min(max(temp_min, base_temp), upper_threshold)
            gdd = (t_max + t_min) / 2 - base_temp
        
        else:
            gdd = max(0, ((temp_max + temp_min) / 2) - base_temp)
        
        return {
            'gdd': round(gdd, 2),
            'base_temperature': base_temp,
            'method': method,
            'effective_temp_max': min(temp_max, upper_threshold),
            'effective_temp_min': max(temp_min, base_temp)
        }
    
    def estimate_growth_stage(
        self,
        accumulated_gdd: float,
        crop: str
    ) -> Dict:
        """Estimate crop growth stage based on accumulated GDD"""
        
        # GDD requirements for growth stages (approximate)
        gdd_stages = {
            'rice': {'emergence': 100, 'tillering': 400, 'flowering': 1200, 'maturity': 2000},
            'wheat': {'emergence': 80, 'tillering': 350, 'flowering': 1000, 'maturity': 1500},
            'maize': {'emergence': 80, 'vegetative': 400, 'flowering': 900, 'maturity': 1400},
            'cotton': {'emergence': 60, 'squaring': 450, 'flowering': 850, 'boll_open': 1600},
            'tomato': {'emergence': 50, 'flowering': 500, 'fruit_set': 800, 'maturity': 1200},
            'potato': {'emergence': 60, 'tuber_init': 400, 'bulking': 800, 'maturity': 1200}
        }
        
        stages = gdd_stages.get(crop.lower(), {
            'emergence': 80, 'vegetative': 400, 'reproductive': 800, 'maturity': 1200
        })
        
        current_stage = 'pre-emergence'
        next_stage = 'emergence'
        progress = 0
        
        sorted_stages = sorted(stages.items(), key=lambda x: x[1])
        
        for i, (stage, gdd_req) in enumerate(sorted_stages):
            if accumulated_gdd >= gdd_req:
                current_stage = stage
                if i + 1 < len(sorted_stages):
                    next_stage = sorted_stages[i + 1][0]
                    next_gdd = sorted_stages[i + 1][1]
                    progress = min(100, (accumulated_gdd - gdd_req) / (next_gdd - gdd_req) * 100)
                else:
                    next_stage = 'harvest'
                    progress = 100
        
        return {
            'current_stage': current_stage,
            'next_stage': next_stage,
            'progress_to_next': round(progress, 1),
            'accumulated_gdd': accumulated_gdd,
            'stages': stages
        }


class YieldPredictor:
    """
    Crop Yield Prediction Model
    
    Uses multiple factors:
    - Weather conditions
    - Soil properties
    - Management practices
    - Historical data patterns
    """
    
    def __init__(self):
        self.base_yields = Config.BASE_YIELDS
        self.yield_factors = Config.YIELD_FACTORS
    
    def predict_yield(
        self,
        crop: str,
        temp_avg: float,
        rainfall: float,
        soil_n: float,
        soil_p: float,
        soil_k: float,
        soil_ph: float,
        irrigation_efficiency: float = 0.7,
        pest_pressure: float = 0.1,
        variety_factor: float = 1.0
    ) -> Dict:
        """
        Predict crop yield based on multiple factors.
        
        Returns yield in kg/hectare with confidence interval.
        """
        base_yield = self.base_yields.get(crop.lower(), 3000)
        
        # Temperature stress factor
        temp_factor = self._calculate_temp_factor(temp_avg, crop)
        
        # Water availability factor
        water_factor = self._calculate_water_factor(rainfall, crop, irrigation_efficiency)
        
        # Nutrient availability factor
        nutrient_factor = self._calculate_nutrient_factor(soil_n, soil_p, soil_k, crop)
        
        # Soil pH factor
        ph_factor = self._calculate_ph_factor(soil_ph)
        
        # Pest/disease pressure factor
        pest_factor = max(0.5, 1 - pest_pressure)
        
        # Combined yield calculation
        predicted_yield = base_yield * temp_factor * water_factor * nutrient_factor * ph_factor * pest_factor * variety_factor
        
        # Calculate confidence based on factor variability
        factors = [temp_factor, water_factor, nutrient_factor, ph_factor, pest_factor]
        factor_variance = np.var(factors)
        confidence = max(50, min(95, 85 - factor_variance * 100))
        
        # Yield range (±15%)
        yield_min = predicted_yield * 0.85
        yield_max = predicted_yield * 1.15
        
        return {
            'predicted_yield': round(predicted_yield, 0),
            'yield_min': round(yield_min, 0),
            'yield_max': round(yield_max, 0),
            'confidence': round(confidence, 1),
            'unit': 'kg/ha',
            'factors': {
                'temperature': round(temp_factor, 3),
                'water': round(water_factor, 3),
                'nutrients': round(nutrient_factor, 3),
                'soil_ph': round(ph_factor, 3),
                'pest_pressure': round(pest_factor, 3)
            },
            'limiting_factor': self._identify_limiting_factor(
                temp_factor, water_factor, nutrient_factor, ph_factor, pest_factor
            )
        }
    
    def _calculate_temp_factor(self, temp: float, crop: str) -> float:
        optimal = self.yield_factors['temperature_stress']['optimal']
        penalty = self.yield_factors['temperature_stress']['penalty_per_degree']
        
        if optimal[0] <= temp <= optimal[1]:
            return 1.0
        elif temp < optimal[0]:
            return max(0.3, 1 - (optimal[0] - temp) * penalty)
        else:
            return max(0.3, 1 - (temp - optimal[1]) * penalty)
    
    def _calculate_water_factor(self, rainfall: float, crop: str, irrigation_eff: float) -> float:
        water_needs = Config.CROP_WATER_NEEDS.get(crop.lower(), 500)
        water_ratio = (rainfall * (1 + irrigation_eff)) / water_needs
        
        if water_ratio >= 1.0:
            return min(1.0, 1.1 - 0.1 * (water_ratio - 1))  # Slight penalty for excess
        else:
            return max(0.3, water_ratio ** 0.7)  # Non-linear response
    
    def _calculate_nutrient_factor(self, n: float, p: float, k: float, crop: str) -> float:
        fert_req = Config.CROP_FERTILIZER.get(crop.lower(), {'N': 100, 'P': 50, 'K': 50})
        
        n_ratio = min(1.2, n / max(fert_req['N'], 1))
        p_ratio = min(1.2, p / max(fert_req['P'], 1))
        k_ratio = min(1.2, k / max(fert_req['K'], 1))
        
        # Liebig's Law of Minimum
        return min(n_ratio, p_ratio, k_ratio)
    
    def _calculate_ph_factor(self, ph: float) -> float:
        optimal = self.yield_factors['soil_ph']
        if optimal['optimal'][0] <= ph <= optimal['optimal'][1]:
            return 1.0
        elif ph < optimal['optimal'][0]:
            return max(0.5, 1 - (optimal['optimal'][0] - ph) * optimal['penalty_per_unit'])
        else:
            return max(0.5, 1 - (ph - optimal['optimal'][1]) * optimal['penalty_per_unit'])
    
    def _identify_limiting_factor(self, temp, water, nutrient, ph, pest) -> str:
        factors = {
            'temperature': temp,
            'water': water,
            'nutrients': nutrient,
            'soil_ph': ph,
            'pest_pressure': pest
        }
        return min(factors, key=factors.get)


class DiseaseRiskPredictor:
    """
    Plant Disease and Pest Risk Prediction Model
    
    Based on:
    - Weather conditions (temp, humidity, rain)
    - Historical disease pressure
    - Crop-specific susceptibility
    """
    
    def __init__(self):
        self.disease_models = Config.DISEASE_MODELS
    
    def calculate_disease_risk(
        self,
        crop: str,
        temp: float,
        humidity: float,
        rainfall: float,
        dew_hours: float = 0,
        consecutive_wet_days: int = 0
    ) -> Dict:
        """
        Calculate disease risk for a crop based on weather conditions.
        """
        risks = {}
        crop_lower = crop.lower()
        
        for disease, params in self.disease_models.items():
            if crop_lower in params['crops']:
                risk = self._calculate_single_disease_risk(
                    disease, params, temp, humidity, rainfall, dew_hours, consecutive_wet_days
                )
                risks[disease] = risk
        
        # Overall risk
        if risks:
            max_risk = max(risks.values(), key=lambda x: x['risk_score'])
            overall_risk = max_risk['risk_score']
            primary_threat = max_risk['disease']
        else:
            overall_risk = 0
            primary_threat = None
        
        return {
            'overall_risk': round(overall_risk, 1),
            'risk_level': self._get_risk_level(overall_risk),
            'primary_threat': primary_threat,
            'disease_risks': risks,
            'recommendations': self._get_recommendations(risks, crop)
        }
    
    def _calculate_single_disease_risk(
        self,
        disease: str,
        params: Dict,
        temp: float,
        humidity: float,
        rainfall: float,
        dew_hours: float,
        wet_days: int
    ) -> Dict:
        risk_score = 0
        factors = []
        
        # Temperature factor
        temp_range = params.get('temp_range', (15, 30))
        if temp_range[0] <= temp <= temp_range[1]:
            temp_factor = 1 - abs(temp - (temp_range[0] + temp_range[1]) / 2) / (temp_range[1] - temp_range[0])
            risk_score += temp_factor * 30
            factors.append(f"Temperature in favorable range ({temp_range[0]}-{temp_range[1]}°C)")
        
        # Humidity factor
        if 'humidity_min' in params:
            if humidity >= params['humidity_min']:
                hum_factor = min(1, (humidity - params['humidity_min']) / 20)
                risk_score += hum_factor * 35
                factors.append(f"High humidity ({humidity}%)")
        
        if 'humidity_range' in params:
            h_range = params['humidity_range']
            if h_range[0] <= humidity <= h_range[1]:
                risk_score += 30
                factors.append(f"Humidity in risk range ({h_range[0]}-{h_range[1]}%)")
        
        # Rain trigger
        if 'rain_trigger' in params and rainfall >= params['rain_trigger']:
            risk_score += 20
            factors.append(f"Rainfall above trigger ({params['rain_trigger']}mm)")
        
        # Dew hours
        if 'dew_hours' in params and dew_hours >= params['dew_hours']:
            risk_score += 15
            factors.append(f"Extended dew period ({dew_hours}hrs)")
        
        # Consecutive wet days bonus
        if wet_days >= 3:
            risk_score += min(15, wet_days * 3)
            factors.append(f"{wet_days} consecutive wet days")
        
        return {
            'disease': disease,
            'risk_score': min(100, risk_score),
            'contributing_factors': factors
        }
    
    def _get_risk_level(self, score: float) -> str:
        if score >= 70:
            return 'high'
        elif score >= 40:
            return 'moderate'
        elif score >= 20:
            return 'low'
        else:
            return 'minimal'
    
    def _get_recommendations(self, risks: Dict, crop: str) -> List[str]:
        recommendations = []
        
        for disease, risk_data in risks.items():
            if risk_data['risk_score'] >= 40:
                if disease == 'late_blight':
                    recommendations.append("Apply preventive fungicide (Mancozeb/Metalaxyl)")
                    recommendations.append("Avoid overhead irrigation")
                elif disease == 'powdery_mildew':
                    recommendations.append("Apply sulfur-based fungicide")
                    recommendations.append("Improve air circulation")
                elif disease == 'rust':
                    recommendations.append("Scout for early symptoms")
                    recommendations.append("Consider Propiconazole application")
                elif disease == 'bacterial_leaf_blight':
                    recommendations.append("Reduce nitrogen application")
                    recommendations.append("Maintain field drainage")
        
        if not recommendations:
            recommendations.append("No immediate action required. Continue monitoring.")
        
        return recommendations[:4]  # Limit to 4 recommendations


class SoilHealthAnalyzer:
    """
    Soil Health Index (SHI) Calculator
    
    Composite index based on:
    - Chemical properties (NPK, pH, EC)
    - Physical properties (texture)
    - Biological indicators (organic carbon)
    """
    
    def __init__(self):
        self.weights = Config.SOIL_HEALTH_WEIGHTS
        self.optimal = Config.OPTIMAL_SOIL
    
    def calculate_soil_health_index(
        self,
        nitrogen: float,
        phosphorus: float,
        potassium: float,
        ph: float,
        organic_carbon: float = None,
        ec: float = None,
        soil_type: str = 'loam'
    ) -> Dict:
        """
        Calculate comprehensive Soil Health Index (0-100).
        """
        scores = {}
        
        # Nitrogen score
        scores['nitrogen'] = self._calculate_param_score(
            nitrogen,
            self.optimal['nitrogen']['min'],
            self.optimal['nitrogen']['max'],
            self.optimal['nitrogen']['ideal']
        )
        
        # Phosphorus score
        scores['phosphorus'] = self._calculate_param_score(
            phosphorus,
            self.optimal['phosphorus']['min'],
            self.optimal['phosphorus']['max'],
            self.optimal['phosphorus']['ideal']
        )
        
        # Potassium score
        scores['potassium'] = self._calculate_param_score(
            potassium,
            self.optimal['potassium']['min'],
            self.optimal['potassium']['max'],
            self.optimal['potassium']['ideal']
        )
        
        # pH score
        scores['ph_balance'] = self._calculate_ph_score(ph)
        
        # Organic carbon score (estimated if not provided)
        if organic_carbon is None:
            organic_carbon = self._estimate_organic_carbon(soil_type, nitrogen)
        scores['organic_carbon'] = self._calculate_param_score(
            organic_carbon,
            self.optimal['organic_carbon']['min'],
            self.optimal['organic_carbon']['max'],
            self.optimal['organic_carbon']['ideal']
        )
        
        # EC score
        if ec is not None:
            scores['ec'] = self._calculate_param_score(
                ec,
                self.optimal['ec']['min'],
                self.optimal['ec']['max'],
                self.optimal['ec']['ideal'],
                inverse=True  # Lower is better for EC
            )
        else:
            scores['ec'] = 70  # Assume moderate
        
        # Texture score
        scores['texture'] = self._get_texture_score(soil_type)
        
        # Weighted composite index
        shi = sum(
            scores[param] * self.weights[param]
            for param in self.weights
            if param in scores
        )
        
        return {
            'soil_health_index': round(shi, 1),
            'rating': self._get_rating(shi),
            'component_scores': {k: round(v, 1) for k, v in scores.items()},
            'limiting_factors': self._identify_limiting_factors(scores),
            'recommendations': self._generate_recommendations(scores, soil_type)
        }
    
    def _calculate_param_score(
        self,
        value: float,
        min_val: float,
        max_val: float,
        ideal: float,
        inverse: bool = False
    ) -> float:
        if inverse:
            if value <= ideal:
                return 100
            elif value >= max_val:
                return 30
            else:
                return 100 - 70 * (value - ideal) / (max_val - ideal)
        
        if min_val <= value <= max_val:
            # Closer to ideal = higher score
            distance = abs(value - ideal) / max(ideal - min_val, max_val - ideal)
            return 100 - distance * 30
        elif value < min_val:
            return max(20, 100 - (min_val - value) / min_val * 80)
        else:
            return max(20, 100 - (value - max_val) / max_val * 80)
    
    def _calculate_ph_score(self, ph: float) -> float:
        ideal = self.optimal['ph']['ideal']
        if 6.0 <= ph <= 7.5:
            return 100 - abs(ph - ideal) * 10
        elif 5.5 <= ph < 6.0 or 7.5 < ph <= 8.0:
            return 60
        else:
            return 30
    
    def _estimate_organic_carbon(self, soil_type: str, nitrogen: float) -> float:
        """Estimate OC from soil type and N content (Walkley-Black correlation)"""
        type_factors = {
            'alluvial': 0.6, 'black': 0.8, 'red': 0.4, 'laterite': 0.3,
            'forest': 1.2, 'peaty': 2.0, 'saline': 0.3, 'arid': 0.2, 'loam': 0.7
        }
        base = type_factors.get(soil_type.lower(), 0.6)
        return base + (nitrogen / 1000) * 0.3
    
    def _get_texture_score(self, soil_type: str) -> float:
        scores = {
            'loam': 95, 'alluvial': 90, 'black': 85, 'red': 75,
            'forest': 80, 'laterite': 60, 'saline': 40, 'arid': 45, 'peaty': 70
        }
        return scores.get(soil_type.lower(), 70)
    
    def _get_rating(self, shi: float) -> str:
        if shi >= 80:
            return 'Excellent'
        elif shi >= 65:
            return 'Good'
        elif shi >= 50:
            return 'Moderate'
        elif shi >= 35:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _identify_limiting_factors(self, scores: Dict) -> List[str]:
        factors = []
        for param, score in scores.items():
            if score < 50:
                factors.append(param.replace('_', ' ').title())
        return factors
    
    def _generate_recommendations(self, scores: Dict, soil_type: str) -> List[str]:
        recs = []
        
        if scores.get('nitrogen', 100) < 60:
            recs.append("Apply nitrogen-rich fertilizers (Urea, Ammonium Sulfate)")
        if scores.get('phosphorus', 100) < 60:
            recs.append("Apply phosphatic fertilizers (DAP, SSP)")
        if scores.get('potassium', 100) < 60:
            recs.append("Apply potash fertilizers (MOP, SOP)")
        if scores.get('organic_carbon', 100) < 60:
            recs.append("Incorporate organic matter (FYM, compost, green manure)")
        if scores.get('ph_balance', 100) < 60:
            recs.append("Apply lime (acidic soil) or gypsum (alkaline soil)")
        if scores.get('ec', 100) < 50:
            recs.append("Improve drainage and leach excess salts")
        
        if not recs:
            recs.append("Soil health is good. Maintain current practices.")
        
        return recs[:4]


class IrrigationScheduler:
    """
    Smart Irrigation Scheduling based on:
    - Crop water requirements
    - Soil moisture balance
    - Weather forecast
    - Growth stage
    """
    
    def __init__(self):
        self.et_calc = EvapotranspirationCalculator()
        self.gdd_calc = GrowingDegreeDaysCalculator()
    
    def calculate_irrigation_schedule(
        self,
        crop: str,
        area_hectares: float,
        soil_type: str,
        current_soil_moisture: float,  # %
        weather_forecast: List[Dict],  # 7-day forecast
        planting_date: datetime = None,
        irrigation_system: str = 'flood'  # flood, drip, sprinkler
    ) -> Dict:
        """
        Generate irrigation schedule for the next 7 days.
        """
        efficiency = {'flood': 0.5, 'sprinkler': 0.75, 'drip': 0.9}.get(irrigation_system, 0.6)
        
        # Soil water holding capacity (mm/m) by type
        whc = {
            'alluvial': 150, 'black': 200, 'red': 100, 'loam': 170,
            'laterite': 80, 'saline': 120, 'arid': 60, 'forest': 180, 'peaty': 250
        }.get(soil_type.lower(), 150)
        
        # Root zone depth (m) - simplified
        root_depth = 0.5
        total_available_water = whc * root_depth  # mm
        
        # Management Allowable Depletion (MAD) - typically 50%
        mad = 0.5
        allowable_depletion = total_available_water * mad
        
        schedule = []
        soil_moisture_mm = current_soil_moisture / 100 * total_available_water
        accumulated_gdd = 0
        
        for day_data in weather_forecast[:7]:
            temp_max = day_data.get('temp_max', 30)
            temp_min = day_data.get('temp_min', 20)
            humidity = day_data.get('humidity', 60)
            wind = day_data.get('wind_speed', 2)
            rainfall = day_data.get('rainfall', 0)
            
            # Calculate ET
            et0_data = self.et_calc.calculate_et0(
                temp_max, temp_min, humidity, wind
            )
            et0 = et0_data['et0']
            
            # Get growth stage for Kc
            gdd_day = self.gdd_calc.calculate_gdd(temp_max, temp_min, crop)['gdd']
            accumulated_gdd += gdd_day
            stage_info = self.gdd_calc.estimate_growth_stage(accumulated_gdd, crop)
            growth_stage = stage_info['current_stage']
            
            # Map growth stage to Kc stage
            kc_stage = 'mid'
            if 'emergence' in growth_stage or 'initial' in growth_stage:
                kc_stage = 'initial'
            elif 'maturity' in growth_stage or 'late' in growth_stage:
                kc_stage = 'late'
            
            etc = self.et_calc.calculate_crop_et(et0, crop, kc_stage)
            
            # Water balance
            soil_moisture_mm += rainfall
            soil_moisture_mm -= etc
            soil_moisture_mm = max(0, min(soil_moisture_mm, total_available_water))
            
            # Check if irrigation needed
            irrigation_needed = 0
            if soil_moisture_mm < (total_available_water - allowable_depletion):
                irrigation_needed = (total_available_water * 0.9 - soil_moisture_mm) / efficiency
                soil_moisture_mm = total_available_water * 0.9
            
            schedule.append({
                'date': day_data.get('date', ''),
                'et0': et0,
                'etc': etc,
                'rainfall': rainfall,
                'irrigation_mm': round(irrigation_needed, 1),
                'irrigation_liters': round(irrigation_needed * area_hectares * 10000, 0),
                'soil_moisture_percent': round(soil_moisture_mm / total_available_water * 100, 1),
                'growth_stage': growth_stage
            })
        
        total_irrigation = sum(d['irrigation_mm'] for d in schedule)
        
        return {
            'schedule': schedule,
            'total_irrigation_mm': round(total_irrigation, 1),
            'total_irrigation_liters': round(total_irrigation * area_hectares * 10000, 0),
            'irrigation_efficiency': efficiency,
            'system': irrigation_system,
            'recommendations': self._get_irrigation_recommendations(schedule, irrigation_system)
        }
    
    def _get_irrigation_recommendations(self, schedule: List[Dict], system: str) -> List[str]:
        recs = []
        
        high_demand_days = [d for d in schedule if d['etc'] > 5]
        if high_demand_days:
            recs.append(f"High water demand expected on {len(high_demand_days)} days")
        
        if system == 'flood':
            recs.append("Consider switching to drip irrigation to save 40% water")
        
        rainy_days = [d for d in schedule if d['rainfall'] > 5]
        if rainy_days:
            recs.append(f"Skip irrigation on {len(rainy_days)} days due to expected rainfall")
        
        return recs


# Global instances for easy access
et_calculator = EvapotranspirationCalculator()
gdd_calculator = GrowingDegreeDaysCalculator()
yield_predictor = YieldPredictor()
disease_predictor = DiseaseRiskPredictor()
soil_analyzer = SoilHealthAnalyzer()
irrigation_scheduler = IrrigationScheduler()
