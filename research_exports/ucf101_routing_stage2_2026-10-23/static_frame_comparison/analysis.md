# Static Fusion+Frame versus dynamic routers

Original Test used for selection; not independent-test results. Positive ΔNLL = dynamic worse than static.

| Dynamic | Static-only correct | Dynamic-only correct | Changed predictions | ΔNLL |
|---|---:|---:|---:|---:|
| current | 1 | 4 | 6 | 0.00234639 |
| gru | 1 | 4 | 6 | 0.00321388 |
| conditional | 1 | 4 | 5 | 0.00276541 |

## current: changed predictions

- 018_v_BreastStroke_g05_c01: truth=BreastStroke; static=Billiards; dynamic=Skiing; both_wrong
- 041_v_HorseRiding_g23_c04: truth=HorseRiding; static=Rowing; dynamic=HorseRiding; dynamic_only_correct
- 044_v_JavelinThrow_g23_c02: truth=JavelinThrow; static=PoleVault; dynamic=JavelinThrow; dynamic_only_correct
- 077_v_ShavingBeard_g04_c01: truth=ShavingBeard; static=BlowDryHair; dynamic=ShavingBeard; dynamic_only_correct
- 081_v_Skijet_g23_c02: truth=Skijet; static=Skijet; dynamic=Biking; static_only_correct
- 095_v_UnevenBars_g07_c02: truth=UnevenBars; static=StillRings; dynamic=UnevenBars; dynamic_only_correct

NLL attribution (sum and contribution to dataset mean):

```json
{
  "both_correct": {
    "videos": 2457,
    "sum_dynamic_minus_static": -24.735137400046302,
    "contribution_to_mean": -0.009681071389450607
  },
  "both_wrong": {
    "videos": 93,
    "sum_dynamic_minus_static": 30.897522694800177,
    "contribution_to_mean": 0.012092963872720226
  },
  "static_only_correct": {
    "videos": 1,
    "sum_dynamic_minus_static": 0.028462804038337453,
    "contribution_to_mean": 1.1140040719505852e-05
  },
  "dynamic_only_correct": {
    "videos": 4,
    "sum_dynamic_minus_static": -0.19581434575575285,
    "contribution_to_mean": -7.66396656578289e-05
  }
}
```

## gru: changed predictions

- 018_v_BreastStroke_g05_c01: truth=BreastStroke; static=Billiards; dynamic=Skiing; both_wrong
- 041_v_HorseRiding_g23_c04: truth=HorseRiding; static=Rowing; dynamic=HorseRiding; dynamic_only_correct
- 044_v_JavelinThrow_g23_c02: truth=JavelinThrow; static=PoleVault; dynamic=JavelinThrow; dynamic_only_correct
- 077_v_ShavingBeard_g04_c01: truth=ShavingBeard; static=BlowDryHair; dynamic=ShavingBeard; dynamic_only_correct
- 081_v_Skijet_g23_c02: truth=Skijet; static=Skijet; dynamic=Biking; static_only_correct
- 095_v_UnevenBars_g07_c02: truth=UnevenBars; static=StillRings; dynamic=UnevenBars; dynamic_only_correct

NLL attribution (sum and contribution to dataset mean):

```json
{
  "both_correct": {
    "videos": 2457,
    "sum_dynamic_minus_static": -24.246024933765675,
    "contribution_to_mean": -0.009489637938851537
  },
  "both_wrong": {
    "videos": 93,
    "sum_dynamic_minus_static": 32.62788514413196,
    "contribution_to_mean": 0.012770209449758105
  },
  "static_only_correct": {
    "videos": 1,
    "sum_dynamic_minus_static": 0.029923443340357014,
    "contribution_to_mean": 1.1711719506989047e-05
  },
  "dynamic_only_correct": {
    "videos": 4,
    "sum_dynamic_minus_static": -0.20031205106965744,
    "contribution_to_mean": -7.840001998812424e-05
  }
}
```

## conditional: changed predictions

- 041_v_HorseRiding_g23_c04: truth=HorseRiding; static=Rowing; dynamic=HorseRiding; dynamic_only_correct
- 044_v_JavelinThrow_g23_c02: truth=JavelinThrow; static=PoleVault; dynamic=JavelinThrow; dynamic_only_correct
- 077_v_ShavingBeard_g04_c01: truth=ShavingBeard; static=BlowDryHair; dynamic=ShavingBeard; dynamic_only_correct
- 081_v_Skijet_g23_c02: truth=Skijet; static=Skijet; dynamic=Biking; static_only_correct
- 095_v_UnevenBars_g07_c02: truth=UnevenBars; static=StillRings; dynamic=UnevenBars; dynamic_only_correct

NLL attribution (sum and contribution to dataset mean):

```json
{
  "both_correct": {
    "videos": 2457,
    "sum_dynamic_minus_static": -24.179220764348404,
    "contribution_to_mean": -0.009463491492895658
  },
  "both_wrong": {
    "videos": 93,
    "sum_dynamic_minus_static": 31.429062680667332,
    "contribution_to_mean": 0.012301003006132029
  },
  "static_only_correct": {
    "videos": 1,
    "sum_dynamic_minus_static": 0.016359527330375156,
    "contribution_to_mean": 6.402946117563662e-06
  },
  "dynamic_only_correct": {
    "videos": 4,
    "sum_dynamic_minus_static": -0.20058715749183675,
    "contribution_to_mean": -7.850769373457407e-05
  }
}
```
