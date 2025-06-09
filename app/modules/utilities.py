class Utilities:
    @staticmethod
    def add_missing_from_other(target: dict, source: dict) -> dict:
        """
        Adds only missing key-value pairs from source to target dict.
        Existing keys in target are not overwritten.
        Returns the updated target dict.
        """
        for key, value in source.items():
            if key not in target:
                target[key] = value
        return target
