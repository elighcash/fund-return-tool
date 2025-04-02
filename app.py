from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
import os
from models.pe_fund import PEFund

app = Flask(__name__)

# Load the fund data
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fund-data')
def fund_data():
    # Initialize PE fund model with data
    fund = PEFund('data/fund_data.csv')
    
    # Get parameters from the request - UPDATE DEFAULTS
    loc_rate = float(request.args.get('loc_rate', 6)) / 100  # Default 6%
    loc_delay = int(request.args.get('loc_delay', 2))        # Default 2q
    
    nav_rate = float(request.args.get('nav_rate', 8)) / 100  # Default 8%
    nav_pct = float(request.args.get('nav_pct', 30))         # Default 30%
    nav_term = float(request.args.get('nav_term', 3))        # Default 3 years
    nav_origination_year = int(request.args.get('nav_origination_year', 3)) # Default Year 3 (unchanged)
    
    print(f"Parameters: LOC ({loc_rate:.2%}, {loc_delay}q), NAV ({nav_rate:.2%}, {nav_pct:.0f}%, {nav_term}y, Year {nav_origination_year})")
    
    # Calculate base IRR
    base_irr = fund.calculate_irr()
    
    if base_irr is None:
        print("ERROR: IRR calculation failed completely")
        return jsonify({"error": "Failed to calculate IRR"}), 500
    
    print(f"Base IRR calculated: {base_irr*100:.2f}%")
    
    # Apply LOC strategy
    loc_data, loc_loans, loc_interest = fund.apply_loc(loc_rate, loc_delay)
    loc_fund = PEFund.__new__(PEFund)
    loc_fund.data = loc_data
    loc_irr = loc_fund.calculate_irr()
    
    # Fix the f-string formatting issue with conditionals
    if loc_irr is not None:
        print(f"LOC IRR calculated: {loc_irr*100:.2f}%")
    else:
        print("LOC IRR calculation failed")
    
    # Apply NAV loan strategy
    nav_data, nav_loans, nav_interest = fund.apply_nav_loan(nav_rate, nav_pct, nav_term, nav_origination_year) # Pass new param
    nav_fund = PEFund.__new__(PEFund)
    nav_fund.data = nav_data
    nav_irr = nav_fund.calculate_irr()
    
    if nav_irr is not None:
        print(f"NAV IRR calculated: {nav_irr*100:.2f}%")
    else:
        print("NAV IRR calculation failed")
    
    # Apply combined strategy
    combined_data, combined_loans, combined_interest = fund.apply_combined(
        loc_rate, loc_delay, nav_rate, nav_pct, nav_term, nav_origination_year
    )
    combined_fund = PEFund.__new__(PEFund)
    combined_fund.data = combined_data
    combined_irr = combined_fund.calculate_irr()
    
    if combined_irr is not None:
        print(f"Combined IRR calculated: {combined_irr*100:.2f}%")
    else:
        print("Combined IRR calculation failed")
    
    # Calculate totals for each scenario
    def calculate_totals(data):
        total_contrib = data['Contribution'].sum()
        total_distrib = data['Distribution'].sum()
        # Adjust total distributions for any interest paid if tracked separately
        # total_distrib -= data['Interest Paid'].sum() if 'Interest Paid' in data else 0
        final_nav = data['NAV'].iloc[-1]
        total_value = total_distrib + final_nav
        return {
            'contributions': float(total_contrib),
            'distributions': float(total_distrib),
            'nav': float(final_nav),
            'total_value': float(total_value),
            'multiple': float(total_value / total_contrib) if total_contrib > 0 else 0
        }
    
    base_totals = calculate_totals(fund.data)
    loc_totals = calculate_totals(loc_data)
    nav_totals = calculate_totals(nav_data)
    combined_totals = calculate_totals(combined_data)
    
    # --- Calculate Net Cash Flows for each scenario ---
    def calculate_net_cf(data):
        # Net CF = Distribution - Contribution
        net_cf = (data['Distribution'] - data['Contribution']).tolist()
        # Add final NAV to the last period's cash flow
        if len(net_cf) > 0 and 'NAV' in data and len(data['NAV']) > 0:
             # Ensure NAV is treated as float, handle potential NaNs
             final_nav = float(data['NAV'].iloc[-1]) if pd.notna(data['NAV'].iloc[-1]) else 0.0
             net_cf[-1] += final_nav
        return net_cf

    base_net_cf = calculate_net_cf(fund.data)
    loc_net_cf = calculate_net_cf(loc_data)
    nav_net_cf = calculate_net_cf(nav_data)
    combined_net_cf = calculate_net_cf(combined_data)
    # --- End Net Cash Flow Calculation ---

    # Prepare chart data with all scenarios
    chart_data = {
        'dates': fund.data['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'base': {
            'contributions': (-fund.data['Contribution']).tolist(),
            'distributions': fund.data['Distribution'].tolist(),
            'nav': fund.data['NAV'].tolist(),
            'irr': round(base_irr * 100, 2) if base_irr is not None else None,
            'loans': [0] * len(fund.data),
            'interest': [0] * len(fund.data),
            'net_cf': base_net_cf,
            'totals': base_totals
        },
        'loc': {
            'contributions': (-loc_data['Contribution']).tolist(),
            'distributions': loc_data['Distribution'].tolist(),
            'nav': loc_data['NAV'].tolist(),
            'irr': round(loc_irr * 100, 2) if loc_irr is not None else None,
            'loans': loc_loans.tolist() if loc_loans is not None else [0] * len(fund.data),
            'interest': loc_interest.tolist() if loc_interest is not None else [0] * len(fund.data),
            'net_cf': loc_net_cf,
            'totals': loc_totals
        },
        'nav_loan': {
            'contributions': (-nav_data['Contribution']).tolist(),
            'distributions': nav_data['Distribution'].tolist(),
            'nav': nav_data['NAV'].tolist(),
            'irr': round(nav_irr * 100, 2) if nav_irr is not None else None,
            'loans': nav_loans.tolist(),
            'interest': nav_interest.tolist(),
            'net_cf': nav_net_cf,
            'totals': nav_totals
        },
        'combined': {
            'contributions': (-combined_data['Contribution']).tolist(),
            'distributions': combined_data['Distribution'].tolist(),
            'nav': combined_data['NAV'].tolist(),
            'irr': round(combined_irr * 100, 2) if combined_irr is not None else None,
            'loans': combined_loans.tolist(),
            'interest': combined_interest.tolist(),
            'net_cf': combined_net_cf,
            'totals': combined_totals
        }
    }
    
    return jsonify(chart_data)

if __name__ == '__main__':
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # If the data file doesn't exist, create it from the table in the request
    if not os.path.exists('data/fund_data.csv'):
        # This is a simplified version based on the image
        # You would need to extract the actual data from the image
        # For now, I'll create a sample that matches the structure
        data = {
            'Qtr': list(range(1, 57)),
            'Yr': [1 + i//4 for i in range(56)],
            'Date': [
                # First few quarters from the image
                '3/31/25', '6/30/25', '9/30/25', '12/31/25',
                '3/31/26', '6/30/26', '9/30/26', '12/31/26',
                # ... and so on, you should extract all dates
            ],
            'Contrib %': [
                # First few values from the image
                3.75, 3.75, 3.75, 3.75,
                6.25, 6.25, 6.25, 6.25,
                # ... and so on
            ],
            'DPI': [
                # First few values from the image
                0.000, 0.000, 0.000, 0.000,
                0.000, 0.000, 0.000, 0.000,
                # ... and so on
            ],
            'RVPI': [
                # First few values from the image
                0.940, 0.940, 0.940, 0.940,
                0.941, 0.942, 0.943, 0.945,
                # ... and so on
            ]
        }
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv('data/fund_data.csv', index=False)
    
    app.run(debug=True) 