
"""
SELECT
m1.type, m2.type,
mr.model_id,
m1.longitude,
m1.latitude,
m1.height AS manual_measure_height,
mr.float_value,
mr.key
FROM measure AS m1
INNER JOIN measure AS m2 ON m1.person_id=m2.person_id AND m1.type='manual' AND m2.type!='manual'
INNER JOIN measure_result as mr ON m2.id=mr.measure_id
WHERE mr.model_id LIKE '%height%'
AND m1.longitude IS NOT NULL
AND m2.latitude IS NOT NULL
AND mr.model_id = 'GAPNet_height_s1'
-- 26917
"""
# WHERE person_id = '90c909daf9c50584_person_1575358107344_rbsD5MSPHZuV2NDM'
# idea: GROUP BY key
# measures_with_results view
