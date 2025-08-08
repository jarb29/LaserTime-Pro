"""
Cutting Speed Validation Service
Validates ML predictions against physical cutting parameters
"""

import math
from typing import Dict, Any

# Laser cutting parameters lookup table
CUT_VALUES = {
    6: {'nozzle': 'S2.5', 'VC Est (mt/min)': 15, 'cutting height (mm)': 0.6, 'gas': 'Air', 'pressure bar': 8,
        'power %': 100, 'focus': 1.5, 'duty %': 100, 'Frequency': 5000},
    8: {'nozzle': 'S2.5', 'VC Est (mt/min)': 9, 'cutting height (mm)': 0.6, 'gas': 'Air', 'pressure bar': 8,
        'power %': 100, 'focus': 1, 'duty %': 100, 'Frequency': 5000},
    10: {'nozzle': 'S3.0', 'VC Est (mt/min)': 5.5, 'cutting height (mm)': 0.6, 'gas': 'Air', 'pressure bar': 10,
         'power %': 100, 'focus': -1, 'duty %': 100, 'Frequency': 5000},
    12: {'nozzle': 'D1.2E', 'VC Est (mt/min)': 1.2, 'cutting height (mm)': 1, 'gas': 'Air', 'pressure bar': 0.65,
         'power %': 55, 'focus': 11.5, 'duty %': 100, 'Frequency': 5000},
    14: {'nozzle': 'D1.6E', 'VC Est (mt/min)': 1.8, 'cutting height (mm)': 0.8, 'gas': 'O2', 'pressure bar': 0.55,
         'power %': 80, 'focus': 18, 'duty %': 100, 'Frequency': 5000},
    16: {'nozzle': 'D1.6E', 'VC Est (mt/min)': 1.7, 'cutting height (mm)': 0.6, 'gas': 'O2', 'pressure bar': 0.6,
         'power %': 80, 'focus': 18, 'duty %': 100, 'Frequency': 5000},
    20: {'nozzle': 'D1.6E', 'VC Est (mt/min)': 1.35, 'cutting height (mm)': 0.8, 'gas': 'O2', 'pressure bar': 0.62,
         'power %': 100, 'focus': 18, 'duty %': 100, 'Frequency': 5000},
    25: {'nozzle': 'S1.5SP', 'VC Est (mt/min)': 1, 'cutting height (mm)': 0.3, 'gas': 'O2', 'pressure bar': 0.5,
         'power %': 100, 'focus': 19, 'duty %': 100, 'Frequency': 5000},
    30: {'nozzle': 'S1.4SP', 'VC Est (mt/min)': 0.4, 'cutting height (mm)': 0.3, 'gas': 'O2', 'pressure bar': 0.9,
         'power %': 100, 'focus': 19, 'duty %': 100, 'Frequency': 5000},
    35: {'nozzle': 'D1.8E', 'VC Est (mt/min)': 0.3, 'cutting height (mm)': 1, 'gas': 'O2', 'pressure bar': 1.2,
         'power %': 100, 'focus': 20, 'duty %': 100, 'Frequency': 5000},
    40: {'nozzle': 'D1.8E', 'VC Est (mt/min)': 0.25, 'cutting height (mm)': 0.8, 'gas': 'O2', 'pressure bar': 0.93,
         'power %': 100, 'focus': 21, 'duty %': 100, 'Frequency': 5000},
    45: {'nozzle': 'D1.8E', 'VC Est (mt/min)': 0.18, 'cutting height (mm)': 2, 'gas': 'O2', 'pressure bar': 1.15,
         'power %': 100, 'focus': 22, 'duty %': 100, 'Frequency': 5000},
    50: {'nozzle': 'D1.8E', 'VC Est (mt/min)': 0.15, 'cutting height (mm)': 2.2, 'gas': 'O2', 'pressure bar': 1.4,
         'power %': 100, 'focus': 23, 'duty %': 100, 'Frequency': 5000}
}


class ValidationService:
    """Service for validating ML predictions with cutting speed parameters"""

    def get_cutting_speed(self, thickness: float) -> float:
        """Get cutting speed with interpolation for unknown thicknesses"""
        thickness = int(thickness)

        # Direct match
        if thickness in CUT_VALUES:
            return CUT_VALUES[thickness]['VC Est (mt/min)']

        # Find closest values for interpolation
        thicknesses = sorted(CUT_VALUES.keys())
        lower_thickness = None
        higher_thickness = None

        for t in thicknesses:
            if t < thickness:
                lower_thickness = t
            elif t > thickness:
                higher_thickness = t
                break

        # Handle edge cases
        if lower_thickness is None:
            return CUT_VALUES[higher_thickness]['VC Est (mt/min)'] if higher_thickness else 0
        if higher_thickness is None:
            return CUT_VALUES[lower_thickness]['VC Est (mt/min)']

        # Interpolate
        lower_speed = CUT_VALUES[lower_thickness]['VC Est (mt/min)']
        higher_speed = CUT_VALUES[higher_thickness]['VC Est (mt/min)']
        return (lower_speed + higher_speed) / 2

    def get_cutting_parameters(self, thickness: float) -> Dict[str, Any]:
        """Get all cutting parameters for a thickness"""
        thickness = int(thickness)

        if thickness in CUT_VALUES:
            return CUT_VALUES[thickness].copy()

        # For interpolated values, return closest match parameters
        thicknesses = sorted(CUT_VALUES.keys())
        closest = min(thicknesses, key=lambda x: abs(x - thickness))
        params = CUT_VALUES[closest].copy()
        params['VC Est (mt/min)'] = self.get_cutting_speed(thickness)
        params['interpolated'] = True
        return params

    def validate_prediction(self, espesor: float, cutting_length: float,
                          ml_prediction: float, enable_validation: bool = True) -> Dict[str, Any]:
        """
        Validate ML prediction against cutting speed calculations

        Args:
            espesor: Material thickness in mm
            cutting_length: Cutting length in meters
            ml_prediction: ML model prediction in minutes
            enable_validation: Whether to apply validation

        Returns:
            Dictionary with validation results
        """
        try:
            # Get cutting parameters
            cutting_speed = self.get_cutting_speed(espesor)
            cutting_params = self.get_cutting_parameters(espesor)

            # Calculate physics-based estimate (convert to minutes)
            estimated_time = (cutting_length / cutting_speed + cutting_length/100) if cutting_speed > 0 else 0

            if not enable_validation:
                return {
                    'ml_prediction': ml_prediction,
                    'speed_estimate': estimated_time,
                    'validated_result': ml_prediction,
                    'cutting_speed': cutting_speed,
                    'cutting_params': cutting_params,
                    'validation_enabled': False,
                    'status': 'ML Only'
                }

            # Apply validation logic from PyCaret code
            base_time = ml_prediction + (cutting_length / 100)*60

            if estimated_time > 0:
                final_time = max(base_time, estimated_time)
            else:
                final_time = base_time

            validated_result = math.ceil(final_time)

            return {
                'ml_prediction': ml_prediction,
                'speed_estimate': estimated_time,
                'validated_result': validated_result,
                'cutting_speed': cutting_speed,
                'cutting_params': cutting_params,
                'adjustment': adjustment,
                'validation_enabled': True,
                'status': 'Validated' if cutting_speed > 0 else 'ML Only'
            }

        except Exception as e:
            return {
                'ml_prediction': ml_prediction,
                'speed_estimate': 0,
                'validated_result': ml_prediction,
                'cutting_speed': 0,
                'cutting_params': {},
                'validation_enabled': enable_validation,
                'status': 'Error',
                'error': str(e)
            }