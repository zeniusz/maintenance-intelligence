# How it works

## The problem
An operator of oil, gas and energy plants may have thousands of assets sending readings every hour. Nobody can watch every chart, and a fixed alarm limit (for example "vibration > X") is a poor fit: a healthy large compressor and a healthy small pump have very different normal values, and weather shifts temperatures even when nothing is wrong. The project answers three questions:

1. **Which assets should a technician visit first?**
2. **Why** is each one flagged, in words a technician can check?
3. **How much can we trust that ranking?**

## The pipeline in seven steps
| Step | What happens, in plain words | Why it matters |
|---|---|---|
| 1. Make the data | A program creates 100 pretend machines with six months of hourly readings (temperature, vibration, pressure, power, running hours), including missing values, broken readings and faults hidden on purpose and recorded. | Real plant data is private. Knowing where the faults are lets us check honestly whether the system finds them. |
| 2. Add the weather | Real outdoor temperature and humidity per site from the free Open-Meteo service, plus live conditions. If the internet is down, made-up weather is used. | A hot day makes a machine run warmer without anything being wrong. The app always starts. |
| 3. Tidy up | Remove junk values (like -999), fill gaps of up to 6 hours, attach the weather. | Real sensors glitch. |
| 4. Learn what is normal for each machine | Every machine is compared with its own usual behaviour, with weather taken into account and switched-off hours ignored. | One shared limit would be wrong for both a big compressor and a small pump. |
| 5. Spot unusual hours | An Isolation Forest looks at all sensors together and marks hours that look strange for that machine. It learns from the first part of the data and is tested on the later part. | It finds odd combinations a single limit would miss, and needs no examples of past failures. |
| 6. Score each machine from 0 to 100 | How far readings drift from normal (50%), how often the machine was flagged (30%), time since maintenance (20%), raised for machines that matter more (about 0.64 to 1.0). | A formula an engineer can read and adjust, not a black box. |
| 7. Explain it in a sentence | For example: "Vibration +11.0σ vs own baseline; 7% of hours alerted; 4 d since maintenance; criticality 3/5" (σ means "how many typical steps above normal"). | People act on a warning they understand and can check. |

**No data leakage:** the data is split by time (the first 120 days to fit, the rest to evaluate). Baselines, the weather adjustment, the forest and the alert threshold only ever see the training period, and a test checks it.

## Anomaly detection is not failure prediction
The Isolation Forest says "this hour looks unlike this asset's normal". It does **not** say "this asset will fail in N days". Predicting failures needs labelled failure histories and far more failures than six months provide. The risk score therefore prioritises *inspection*, and the dashboard says so.

## Design decisions
- **Per-asset baselines over global limits**, because "normal" differs per machine.
- **Isolation Forest over a supervised model**, because real plants rarely have enough labelled failures.
- **A hand-weighted risk score over a learned one**, because an expert can review it.
- **A synthetic fallback for weather**, so the project never fails to start.
- **Reported against a simple rule**, so the model's value is measured, not assumed.

## Product decisions
- One question per screen area: the Overview answers "what first?", Asset details answers "why, and is it still happening?".
- Colour only where it carries meaning, always with a word.
- No invented outputs: no recommended actions, urgency labels, failure probabilities or remaining-life estimates.
- Honest about what it is: synthetic data, a ranking index, not a failure prediction.
- Shareable by design: the page address holds the sites, filters, tab and asset.
