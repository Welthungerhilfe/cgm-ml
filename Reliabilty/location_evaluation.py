
"""
SELECT measure.longitude,
       measure.latitude,
	   measure_result.float_value,
	   measure_result.json_value,
	   measure_result.confidence_value
FROM measure_result
INNER JOIN measure ON measure_result.measure_id=measure.id
WHERE measure.longitude IS NOT NULL
AND measure.latitude IS NOT NULL
AND measure_result.model_id='GAPNet_height_s1'
LIMIT 20;
"""

"""
SELECT measure.longitude,
       measure.latitude,
	   measure.height,
	   measure_result.float_value,
	   measure_result.json_value,
	   measure_result.confidence_value
FROM measure_result
INNER JOIN measure ON measure_result.measure_id=measure.id
WHERE measure.longitude IS NOT NULL
AND measure.latitude IS NOT NULL
AND measure_result.model_id='GAPNet_height_s1'
AND measure.height > 0
LIMIT 20;
"""

"""
SELECT COUNT(DISTINCT measure.id)
FROM measure_result
INNER JOIN measure ON measure_result.measure_id=measure.id
"""
# 65k

"""
SELECT COUNT(DISTINCT measure.id)
FROM measure
"""
# 80k

# This shows that not all the measure objects have a mesure_result
