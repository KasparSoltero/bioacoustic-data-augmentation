import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

# --- Data Definitions ---
# Dataset details (name, size_GB, acquisition_date_object)
# Handle inconsistent date formats and standardize size
data_defs = {
    # 'landcare possum': {'size': 5, 'date': datetime.strptime('07-02-2023', '%d-%m-%Y')},
    # 'landcare stoat': {'size': 15, 'date': datetime.strptime('20-02-2023', '%d-%m-%Y')},
    'Hinewai': {'size': 15, 'date': datetime.strptime('28-03-2023', '%d-%m-%Y')},
    # 'Sierra Nevada': {'size': 3, 'date': datetime.strptime('10-06-2024', '%d-%m-%Y')},
    'Macaulay Tui+Korimako': {'size': 19, 'date': datetime.strptime('18-01-2025', '%d-%m-%Y')},
    'DOC Tier 1': {'size': 185, 'date': datetime.strptime('21-03-2025', '%d-%m-%Y')},
    'Ruru Nohinohi': {'size': 165, 'date': datetime.strptime('09-03-2025', '%d-%m-%Y')}
}

# Taukahara specific data
taukahara_steps = [
    (datetime.strptime('15-05-2023', '%d-%m-%Y'), 32.9),
    (datetime.strptime('15-06-2023', '%d-%m-%Y'), 32.9 + 30.5),
    (datetime.strptime('15-07-2023', '%d-%m-%Y'), 32.9 + 30.5 + 44.5) # = 107.9
]
taukahara_linear_start_date = datetime.strptime('01-09-2024', '%d-%m-%Y')
taukahara_linear_end_date = datetime.strptime('01-01-2025', '%d-%m-%Y')
taukahara_linear_increase_gb = 1.8 * 1000 # 1.8T
taukahara_final_size = taukahara_steps[-1][1] + taukahara_linear_increase_gb

# --- Prepare Time Axis ---
all_dates = set([datetime(2023, 1, 1)]) # Start point
for info in data_defs.values():
    all_dates.add(info['date'])
for date, _ in taukahara_steps:
    all_dates.add(date)
all_dates.add(taukahara_linear_start_date)
all_dates.add(taukahara_linear_end_date)
all_dates.add(datetime(2026, 1, 1)) # End point

# Add points just before step changes for sharper vertical lines
epsilon = timedelta(microseconds=1)
step_dates = set([info['date'] for info in data_defs.values()] + [d for d, _ in taukahara_steps])
for date in step_dates:
    all_dates.add(date - epsilon)

sorted_dates = sorted(list(all_dates))
x_axis_dates = np.array(sorted_dates)

# --- Calculate Data Series for Stacking ---
y_series_data = []
labels = []

# Taukahara series calculation
taukahara_values = []
current_size = 0
step_idx = 0
linear_start_val = taukahara_steps[-1][1]
linear_duration_days = (taukahara_linear_end_date - taukahara_linear_start_date).total_seconds() / (24 * 3600)

for date in sorted_dates:
    # Apply step changes
    while step_idx < len(taukahara_steps) and date >= taukahara_steps[step_idx][0]:
        current_size = taukahara_steps[step_idx][1]
        step_idx += 1

    # Apply linear increase
    if date >= taukahara_linear_start_date and date <= taukahara_linear_end_date:
        if linear_duration_days > 0:
            elapsed_days = (date - taukahara_linear_start_date).total_seconds() / (24 * 3600)
            fraction = elapsed_days / linear_duration_days
            current_size = linear_start_val + (taukahara_linear_increase_gb * fraction)
        else:
             current_size = taukahara_final_size # Handle zero duration case
    elif date > taukahara_linear_end_date:
         current_size = taukahara_final_size # Max size after linear phase

    taukahara_values.append(current_size)

# Add Taukahara first to stack at the bottom/early
y_series_data.append(np.array(taukahara_values))
labels.append('Taukahara')

# Other datasets (ordered roughly by acquisition for better stacking)
ordered_names = [
    'Hinewai',
    'Macaulay Tui+Korimako', 'Ruru Nohinohi', 'DOC Tier 1'
]

for name in ordered_names:
    info = data_defs[name]
    ds_values = np.zeros(len(sorted_dates))
    acquisition_idx = np.searchsorted(sorted_dates, info['date'])
    ds_values[acquisition_idx:] = info['size']
    y_series_data.append(ds_values)
    labels.append(name)


# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 6))

ax.stackplot(x_axis_dates, y_series_data, labels=labels, alpha=0.8)

ax.set_xlabel("Date Acquired")
ax.set_ylabel("Cumulative Data Volume (GB)")
ax.legend(loc='upper left')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7])) # Ticks every 6 months
ax.xaxis.set_minor_locator(mdates.MonthLocator())
# veritcal line at 2023 04 01
ax.axvline(datetime(2023, 4, 1), color='grey', linestyle='--', alpha=0.5)
fig.autofmt_xdate()
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.set_ylim(bottom=0)
ax.margins(x=0.01) # Reduce whitespace at edges
# x limit right to 2025-04-01
ax.set_xlim(left=x_axis_dates[0], right=datetime(2025, 4, 1))

plt.show()