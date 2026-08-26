# UID Audit Report

## Research Evidence
Based on the 1st-place solution in IEEE-CIS Fraud Detection competition, the winning team focused on classifying CLIENTS/CREDIT CARDS rather than treating every transaction independently. They discovered and analyzed UIDs, separated known/unknown/questionable clients, and used client consistency heavily in validation and feature engineering.
Sources: 
- https://www.kaggle.com/competitions/ieee-fraud-detection/writeups/fraudsquad-1st-place-solution-part-2
- https://www.kaggle.com/c/ieee-fraud-detection/discussion/111454

## Candidate UID Definitions
We evaluated the following UID hypotheses:

### UID_A: card1 + addr1 + (DayNumber - D1)
- **Definition**: card1 + addr1 + (DayNumber - D1)
- **Coverage**: 88.6% of transactions have a non-null UID.
- **Unique UIDs**: 177636
- **Singleton percentage**: 58.3% of UIDs appear only once.
- **Mean transactions per UID**: 2.55
- **Median transactions per UID**: 1.00
- **Max transactions per UID**: 388
- **Training UID coverage**: 88.5% of training rows have a UID.
- **Validation UID coverage**: 89.3% of validation rows have a UID.
- **Validation known UID percentage**: 51.1% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 48.9% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 18.3% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 7.254 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 14.930
- **Average D10 std within UID**: 17.177
- **Average D15 std within UID**: 15.390
- **Percentage of UIDs with any D variation**: 31.6% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 2.18
- **Percentage of UIDs with D-tuple collision**: 34.5% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 4.90
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 45.4%
- **Average UID span (days)**: 16.00 (average days between first and last sighting of a UID).
- **First seen DT range**: 86400 to 13391977
- **Last seen DT range**: 86400 to 13391998

### UID_B: card1-6 + (DayNumber - D1) + DeviceInfo
- **Definition**: card1-6 + (DayNumber - D1) + DeviceInfo
- **Coverage**: 20.2% of transactions have a non-null UID.
- **Unique UIDs**: 60888
- **Singleton percentage**: 73.6% of UIDs appear only once.
- **Mean transactions per UID**: 1.70
- **Median transactions per UID**: 1.00
- **Max transactions per UID**: 74
- **Training UID coverage**: 21.4% of training rows have a UID.
- **Validation UID coverage**: 13.5% of validation rows have a UID.
- **Validation known UID percentage**: 12.9% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 87.1% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 13.2% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 1.172 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 2.717
- **Average D10 std within UID**: 0.462
- **Average D15 std within UID**: 2.284
- **Percentage of UIDs with any D variation**: 9.0% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 1.20
- **Percentage of UIDs with D-tuple collision**: 11.3% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 2.99
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 34.4%
- **Average UID span (days)**: 2.31 (average days between first and last sighting of a UID).
- **First seen DT range**: 86506 to 13390883
- **Last seen DT range**: 86506 to 13391775

### UID_C: card1-6 + (DayNumber - D15) + DeviceInfo
- **Definition**: card1-6 + (DayNumber - D15) + DeviceInfo
- **Coverage**: 7.9% of transactions have a non-null UID.
- **Unique UIDs**: 24877
- **Singleton percentage**: 75.1% of UIDs appear only once.
- **Mean transactions per UID**: 1.63
- **Median transactions per UID**: 1.00
- **Max transactions per UID**: 60
- **Training UID coverage**: 8.2% of training rows have a UID.
- **Validation UID coverage**: 6.3% of validation rows have a UID.
- **Validation known UID percentage**: 14.2% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 85.8% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 8.4% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 3.875 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 4.932
- **Average D10 std within UID**: 0.601
- **Average D15 std within UID**: 1.484
- **Percentage of UIDs with any D variation**: 11.8% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 1.21
- **Percentage of UIDs with D-tuple collision**: 12.4% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 1.72
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 26.1%
- **Average UID span (days)**: 2.85 (average days between first and last sighting of a UID).
- **First seen DT range**: 86549 to 13391775
- **Last seen DT range**: 86549 to 13391775

### UID_D: addr1 + addr2 + P_emaildomain
- **Definition**: addr1 + addr2 + P_emaildomain
- **Coverage**: 73.5% of transactions have a non-null UID.
- **Unique UIDs**: 2099
- **Singleton percentage**: 24.5% of UIDs appear only once.
- **Mean transactions per UID**: 179.39
- **Median transactions per UID**: 6.00
- **Max transactions per UID**: 16170
- **Training UID coverage**: 73.5% of training rows have a UID.
- **Validation UID coverage**: 73.7% of validation rows have a UID.
- **Validation known UID percentage**: 99.9% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 0.1% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 45.7% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 70.853 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 85.039
- **Average D10 std within UID**: 77.273
- **Average D15 std within UID**: 89.883
- **Percentage of UIDs with any D variation**: 62.5% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 96.13
- **Percentage of UIDs with D-tuple collision**: 66.0% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 2.02
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 39.2%
- **Average UID span (days)**: 76.02 (average days between first and last sighting of a UID).
- **First seen DT range**: 86401 to 13358260
- **Last seen DT range**: 139418 to 13391998

### UID_E: addr1 + DeviceInfo
- **Definition**: addr1 + DeviceInfo
- **Coverage**: 13.7% of transactions have a non-null UID.
- **Unique UIDs**: 3372
- **Singleton percentage**: 57.5% of UIDs appear only once.
- **Mean transactions per UID**: 20.77
- **Median transactions per UID**: 1.00
- **Max transactions per UID**: 3154
- **Training UID coverage**: 14.7% of training rows have a UID.
- **Validation UID coverage**: 7.5% of validation rows have a UID.
- **Validation known UID percentage**: 94.4% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 5.6% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 31.6% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 12.132 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 13.234
- **Average D10 std within UID**: 10.690
- **Average D15 std within UID**: 13.562
- **Percentage of UIDs with any D variation**: 16.9% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 3.67
- **Percentage of UIDs with D-tuple collision**: 21.4% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 1.68
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 29.9%
- **Average UID span (days)**: 22.27 (average days between first and last sighting of a UID).
- **First seen DT range**: 86506 to 13376715
- **Last seen DT range**: 86506 to 13391775

### UID_F: addr1 + P_emaildomain
- **Definition**: addr1 + P_emaildomain
- **Coverage**: 73.5% of transactions have a non-null UID.
- **Unique UIDs**: 1980
- **Singleton percentage**: 23.1% of UIDs appear only once.
- **Mean transactions per UID**: 190.17
- **Median transactions per UID**: 7.00
- **Max transactions per UID**: 16175
- **Training UID coverage**: 73.5% of training rows have a UID.
- **Validation UID coverage**: 73.7% of validation rows have a UID.
- **Validation known UID percentage**: 99.9% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 0.1% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 48.1% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 75.009 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 90.073
- **Average D10 std within UID**: 81.852
- **Average D15 std within UID**: 95.103
- **Percentage of UIDs with any D variation**: 65.6% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 101.85
- **Percentage of UIDs with D-tuple collision**: 69.3% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 2.02
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 39.2%
- **Average UID span (days)**: 80.06 (average days between first and last sighting of a UID).
- **First seen DT range**: 86401 to 13358260
- **Last seen DT range**: 139418 to 13391998

### UID_G: card1-6 + addr1
- **Definition**: card1-6 + addr1
- **Coverage**: 87.1% of transactions have a non-null UID.
- **Unique UIDs**: 36144
- **Singleton percentage**: 40.1% of UIDs appear only once.
- **Mean transactions per UID**: 12.34
- **Median transactions per UID**: 2.00
- **Max transactions per UID**: 5102
- **Training UID coverage**: 86.8% of training rows have a UID.
- **Validation UID coverage**: 89.3% of validation rows have a UID.
- **Validation known UID percentage**: 95.4% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 4.6% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 44.9% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 36.774 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 44.744
- **Average D10 std within UID**: 45.381
- **Average D15 std within UID**: 49.983
- **Percentage of UIDs with any D variation**: 46.5% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 8.37
- **Percentage of UIDs with D-tuple collision**: 51.5% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 1.00
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 0.0%
- **Average UID span (days)**: 47.52 (average days between first and last sighting of a UID).
- **First seen DT range**: 86401 to 13391876
- **Last seen DT range**: 86506 to 13391998

### UID_H: card1-6 + DeviceInfo
- **Definition**: card1-6 + DeviceInfo
- **Coverage**: 20.2% of transactions have a non-null UID.
- **Unique UIDs**: 23995
- **Singleton percentage**: 58.8% of UIDs appear only once.
- **Mean transactions per UID**: 4.32
- **Median transactions per UID**: 1.00
- **Max transactions per UID**: 2351
- **Training UID coverage**: 21.4% of training rows have a UID.
- **Validation UID coverage**: 13.6% of validation rows have a UID.
- **Validation known UID percentage**: 77.5% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 22.5% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 28.6% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 11.065 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 9.553
- **Average D10 std within UID**: 3.035
- **Average D15 std within UID**: 8.023
- **Percentage of UIDs with any D variation**: 18.5% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 1.83
- **Percentage of UIDs with D-tuple collision**: 23.3% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 1.68
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 29.9%
- **Average UID span (days)**: 21.57 (average days between first and last sighting of a UID).
- **First seen DT range**: 86506 to 13390883
- **Last seen DT range**: 86506 to 13391775

### UID_I: id_19 + id_20 + id_31
- **Definition**: id_19 + id_20 + id_31
- **Coverage**: 24.0% of transactions have a non-null UID.
- **Unique UIDs**: 16227
- **Singleton percentage**: 41.9% of UIDs appear only once.
- **Mean transactions per UID**: 7.56
- **Median transactions per UID**: 2.00
- **Max transactions per UID**: 1147
- **Training UID coverage**: 25.2% of training rows have a UID.
- **Validation UID coverage**: 16.8% of validation rows have a UID.
- **Validation known UID percentage**: 81.5% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 18.5% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 47.6% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 21.029 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 16.962
- **Average D10 std within UID**: 7.284
- **Average D15 std within UID**: 16.282
- **Percentage of UIDs with any D variation**: 28.6% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 2.68
- **Percentage of UIDs with D-tuple collision**: 35.8% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 2.94
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 35.5%
- **Average UID span (days)**: 24.52 (average days between first and last sighting of a UID).
- **First seen DT range**: 86506 to 13388918
- **Last seen DT range**: 86506 to 13391775

### UID_J: P_emaildomain + R_emaildomain
- **Definition**: P_emaildomain + R_emaildomain
- **Coverage**: 21.9% of transactions have a non-null UID.
- **Unique UIDs**: 625
- **Singleton percentage**: 28.8% of UIDs appear only once.
- **Mean transactions per UID**: 179.07
- **Median transactions per UID**: 4.00
- **Max transactions per UID**: 39054
- **Training UID coverage**: 22.9% of training rows have a UID.
- **Validation UID coverage**: 15.8% of validation rows have a UID.
- **Validation known UID percentage**: 99.8% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 0.2% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 63.4% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 38.188 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 50.870
- **Average D10 std within UID**: 36.509
- **Average D15 std within UID**: 53.302
- **Percentage of UIDs with any D variation**: 45.4% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 27.86
- **Percentage of UIDs with D-tuple collision**: 54.6% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 2.05
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 30.6%
- **Average UID span (days)**: 66.85 (average days between first and last sighting of a UID).
- **First seen DT range**: 86549 to 13348280
- **Last seen DT range**: 140427 to 13391775

### UID_K: addr1 + DayNumber
- **Definition**: addr1 + DayNumber
- **Coverage**: 88.8% of transactions have a non-null UID.
- **Unique UIDs**: 9356
- **Singleton percentage**: 8.5% of UIDs appear only once.
- **Mean transactions per UID**: 48.61
- **Median transactions per UID**: 25.00
- **Max transactions per UID**: 778
- **Training UID coverage**: 88.6% of training rows have a UID.
- **Validation UID coverage**: 90.4% of validation rows have a UID.
- **Validation known UID percentage**: 0.0% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 100.0% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 88.8% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 132.660 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 155.143
- **Average D10 std within UID**: 148.983
- **Average D15 std within UID**: 165.572
- **Percentage of UIDs with any D variation**: 88.2% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 30.62
- **Percentage of UIDs with D-tuple collision**: 88.6% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 6.99
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 54.9%
- **Average UID span (days)**: 0.79 (average days between first and last sighting of a UID).
- **First seen DT range**: 86400 to 13371694
- **Last seen DT range**: 91330 to 13391998

### UID_addr1: addr1
- **Definition**: addr1
- **Coverage**: 88.8% of transactions have a non-null UID.
- **Unique UIDs**: 330
- **Singleton percentage**: 37.6% of UIDs appear only once.
- **Mean transactions per UID**: 1378.15
- **Median transactions per UID**: 3.00
- **Max transactions per UID**: 40442
- **Training UID coverage**: 88.6% of training rows have a UID.
- **Validation UID coverage**: 90.4% of validation rows have a UID.
- **Validation known UID percentage**: 100.0% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 0.0% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 49.1% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 50.831 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 65.372
- **Average D10 std within UID**: 34.531
- **Average D15 std within UID**: 64.537
- **Percentage of UIDs with any D variation**: 50.0% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 699.10
- **Percentage of UIDs with D-tuple collision**: 50.6% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 1.00
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 0.0%
- **Average UID span (days)**: 49.05 (average days between first and last sighting of a UID).
- **First seen DT range**: 86400 to 12495710
- **Last seen DT range**: 430886 to 13391998

### UID_card: card1-6
- **Definition**: card1-6
- **Coverage**: 98.2% of transactions have a non-null UID.
- **Unique UIDs**: 13125
- **Singleton percentage**: 26.1% of UIDs appear only once.
- **Mean transactions per UID**: 38.30
- **Median transactions per UID**: 4.00
- **Max transactions per UID**: 12117
- **Training UID coverage**: 98.1% of training rows have a UID.
- **Validation UID coverage**: 98.8% of validation rows have a UID.
- **Validation known UID percentage**: 98.9% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 1.1% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 62.4% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 54.135 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 60.749
- **Average D10 std within UID**: 60.081
- **Average D15 std within UID**: 67.484
- **Percentage of UIDs with any D variation**: 59.3% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 22.07
- **Percentage of UIDs with D-tuple collision**: 65.3% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 1.00
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 0.0%
- **Average UID span (days)**: 68.14 (average days between first and last sighting of a UID).
- **First seen DT range**: 86401 to 13388780
- **Last seen DT range**: 87266 to 13391998

### UID_DeviceInfo: DeviceInfo
- **Definition**: DeviceInfo
- **Coverage**: 20.5% of transactions have a non-null UID.
- **Unique UIDs**: 1687
- **Singleton percentage**: 25.6% of UIDs appear only once.
- **Mean transactions per UID**: 62.10
- **Median transactions per UID**: 4.00
- **Max transactions per UID**: 41888
- **Training UID coverage**: 21.6% of training rows have a UID.
- **Validation UID coverage**: 13.7% of validation rows have a UID.
- **Validation known UID percentage**: 97.3% of validation UIDs were seen in training.
- **Validation unknown UID percentage**: 2.7% of validation UIDs are new (not seen in training).
- **Validation questionable UID percentage**: 63.3% (defined as UIDs with inconsistent stable fields: addr1, addr2, P_emaildomain, R_emaildomain showing multiple distinct values within the UID).
- **Average D1 std within UID**: 27.559 (average standard deviation of D1 values within each UID).
- **Average D4 std within UID**: 35.938
- **Average D10 std within UID**: 4.597
- **Average D15 std within UID**: 24.181
- **Percentage of UIDs with any D variation**: 50.9% (UIDs where at least one of D1, D4, D10, D15 has std > 0).
- **Average distinct D-tuples (D1,D4,D10,D15) per UID**: 10.44
- **Percentage of UIDs with D-tuple collision**: 60.1% (UIDs where more than one distinct combination of D1,D4,D10,D15 occurs).
- **Fragmentation: average UIDs per proxy UID (card1-6+addr1)**: 1.68
- **Percentage of proxy UIDs mapping to multiple candidate UIDs**: 29.9%
- **Average UID span (days)**: 60.16 (average days between first and last sighting of a UID).
- **First seen DT range**: 86506 to 13371306
- **Last seen DT range**: 89444 to 13391775

## Known / Unknown / Questionable Analysis
We define:
- **Known UID**: UID that has been seen in the training data (i.e., appears at least once in the training partition).
- **Unknown UID**: UID that appears for the first time in the validation partition (i.e., not seen in training).
- **Questionable UID**: UID that shows inconsistent stable fields (addr1, addr2, P_emaildomain, R_emaildomain) – having more than one distinct non-null value for any of these fields within the UID's transactions. This suggests the UID may be grouping together multiple distinct clients or an unstable entity.

## Temporal Safety Analysis
All statistics were computed using only the training and validation partitions (no future data beyond validation). The UID definition uses only fields available at transaction time (TransactionDT, card fields, addr1, addr2, P_emaildomain, R_emaildomain, DeviceInfo, DeviceType, D-series). No label information was used to construct the UID.
The analysis respects the temporal constraint that rows sharing the same TransactionDT cannot see each other by training/validation split only; within each partition we approximated known/unknown based on first occurrence in the entire partition (which may leak future within partition). For a more strict analysis, a streaming approach would be needed.

## Recommendation
Based on the analysis, we recommend the following UID for the first controlled experiment:

- **UID**: UID_G: card1-6 + addr1
- **Justification**: Good coverage (87.1%), low questionable rate (44.9%), high validation known UID rate (95.4%), and reasonable temporal consistency (avg D1 std = 36.774).

## Minimal Feature Set for Model D
For the selected UID, we propose to compute the following temporal aggregates (strictly prior):
1. `tb_uid_prior_count` – number of prior transactions with the same UID.
2. `tb_uid_amt_mean` – mean TransactionAmt of prior transactions with the same UID.
3. `tb_uid_amt_std` – standard deviation of TransactionAmt of prior transactions with the same UID.
4. `tb_uid_recency` – seconds since the most recent prior transaction with the same UID (null if none).
5. `tb_uid_amt_zscore` – (TransactionAmt - tb_uid_amt_mean) / tb_uid_amt_std (if std > 0 else 0).

## Features Explicitly Rejected and Why
- **UIDs relying solely on high-cardinality identity fields (e.g., id_01-id_??)**: These are likely hashed features and not meaningful as client identifiers.
- **UIDs that combine too many fields (e.g., all card + addr + device + D-series)**: Leads to extremely high cardinality and many singletons, reducing statistical power.
- **UIDs that use future information**: Any UID that incorporates labels or future transaction data.

## Exact Leakage Tests Required Before Training
Before training Model D, the following tests must pass:
1. Unit tests for the UID feature builder (similar to `tests/test_addr_features.py`) covering:
   - First-seen UID (count=0, recency=null, mean=null, std=0, zscore=0)
   - Repeated UID (correct aggregates)
   - Same-TransactionDT isolation (rows within a batch do not see each other)
   - Missing UID components yield null/defaults
   - Future-row invariance (adding future rows does not affect past features)
   - No label access
   - Deterministic execution
   - Train→validation state continuity (using validation_boundaries)
   - Correct amount mean/std, recency, z-score
   - Zero/one prior observation behavior
2. Ensure that Model A, Model B, Model C, validation_boundaries.py, causal_graph_features.py, and existing TBGF code remain unmodified.
3. Verify that the feature set includes only the five UID temporal aggregates plus all existing Model B features.
4. Confirm that the output directory is isolated (e.g., `artifacts/models/model_d/`).

---
*Generated by UID audit script on 2026-08-26.*
