class MathematicalValidator:
    """Mathematical validation utility"""
    
    def __init__(self):
        self.initialized = True
    
    def validate_array(self, data, name="data"):
        """Validate numerical array"""
        try:
            if not data or len(data) == 0:
                return {'is_valid': False, 'error': 'Empty data'}
            
            # Check if all values are numeric
            numeric_data = [float(x) for x in data]
            
            return {
                'is_valid': True,
                'length': len(numeric_data),
                'min': min(numeric_data),
                'max': max(numeric_data),
                'mean': sum(numeric_data) / len(numeric_data)
            }
        except (ValueError, TypeError) as e:
            return {'is_valid': False, 'error': str(e)}