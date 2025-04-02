import pandas as pd
import numpy as np
from datetime import datetime
from scipy import optimize

class PEFund:
    def __init__(self, data_path):
        try:
            self.data = pd.read_csv(data_path)
            print("Data loaded successfully.")
            print(f"Data shape: {self.data.shape}")
            print(f"Columns: {self.data.columns.tolist()}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return
        
        self.process_data()
        
    def process_data(self):
        """Process the raw fund data."""
        try:
            self.data['Date'] = pd.to_datetime(self.data['Date'])

            # Use contributions directly from input percentages
            # Assuming 'Contrib %' represents the percentage of total commitment called in that period
            # Let's assume a total commitment of 100 for simplicity if not otherwise defined
            # If 'Contrib %' is already the dollar amount, this is fine.
            # If it's a percentage, we might need a fund size variable.
            # For now, let's assume 'Contrib %' *is* the contribution amount for the period.
            self.data['Contribution'] = self.data['Contrib %'] # Renaming for clarity

            # Calculate Cumulative Contributions
            self.data['Cumulative Contribution'] = self.data['Contribution'].cumsum()

            # For distributions, calculate the quarter-over-quarter change in DPI applied to cumulative contribution
            # DPI = Cumulative Distributions / Cumulative Contributions
            # Cumulative Distributions = DPI * Cumulative Contributions
            self.data['Cumulative Distribution'] = self.data['DPI'] * self.data['Cumulative Contribution']
            # Calculate periodic distribution as the change in cumulative distribution
            self.data['Distribution'] = self.data['Cumulative Distribution'].diff().fillna(self.data['Cumulative Distribution'].iloc[0]) # Handle first period

            # NAV = RVPI * Cumulative Contributions
            self.data['NAV'] = self.data['RVPI'] * self.data['Cumulative Contribution']

            # Ensure non-negative distributions and NAV
            self.data['Distribution'] = self.data['Distribution'].clip(lower=0)
            self.data['NAV'] = self.data['NAV'].clip(lower=0)


            print("Data processed successfully with updated NAV/Distribution logic.")
            print("First few rows:")
            print(self.data[['Date', 'Contrib %', 'Cumulative Contribution', 'DPI', 'RVPI', 'Contribution', 'Distribution', 'NAV']].head())
            print("Last few rows:")
            print(self.data[['Date', 'Contrib %', 'Cumulative Contribution', 'DPI', 'RVPI', 'Contribution', 'Distribution', 'NAV']].tail())

            # Check for potential issues
            if self.data['Cumulative Contribution'].iloc[-1] == 0:
                print("Warning: Total cumulative contribution is zero.")
            if self.data['NAV'].isnull().any() or self.data['Distribution'].isnull().any():
                 print("Warning: NaNs detected in NAV or Distribution after processing.")


        except Exception as e:
            print(f"Error processing data: {e}")
            # Add traceback for more detailed debugging if needed
            import traceback
            traceback.print_exc()
        
    def calculate_irr(self):
        """Calculate the fund's IRR based on standard cash flow convention."""
        try:
            # Convert the dates to periods (fractional years) for IRR calculation
            dates = [date.timestamp() for date in self.data['Date']]
            min_date = min(dates)
            periods = [(date - min_date)/(365.25*24*60*60) for date in dates]  # Convert to years

            # Create arrays for cash flows
            outflows = self.data['Contribution'].values
            inflows = self.data['Distribution'].values

            # Combine into a single cash flow array
            cash_flows = []
            for i in range(len(periods)):
                cf = -outflows[i] + inflows[i]
                cash_flows.append(cf)

            # Add final NAV to the last cash flow
            cash_flows[-1] += self.data['NAV'].iloc[-1]

            # Print cash flows for debugging
            print(f"First few cash flows: {cash_flows[:5]}")
            print(f"Last few cash flows: {cash_flows[-5:]}")

            # Use scipy's IRR function which handles this case better
            def npv_func(rate):
                # Handle edge case where rate is -1 causing division by zero
                if rate <= -1.0:
                    return float('inf') # Return a large number to avoid issues with the solver
                return sum(cf / (1 + rate)**(t) for t, cf in zip(periods, cash_flows))

            # Solve for the rate that makes NPV=0
            result = optimize.newton(npv_func, 0.1)  # Use 10% as starting guess
            print(f"Calculated IRR: {result*100:.2f}%")
            return result

        except Exception as e:
            print(f"IRR calculation error: {e}")
            # Fallback using simple numpy irr (assumes equal periods)
            try:
                cash_flows_simple = [-c + d for c, d in zip(self.data['Contribution'], self.data['Distribution'])]
                cash_flows_simple[-1] += self.data['NAV'].iloc[-1]
                irr = np.irr(cash_flows_simple)
                print(f"Backup IRR calculation used: {irr*100:.2f}%")
                return irr
            except Exception as e2:
                print(f"Backup IRR calculation also failed: {e2}")
                return None

    def apply_loc(self, interest_rate, delay_quarters=4):
        """Apply a line of credit that delays capital calls, tracks the cumulative
        loan balance, and draws interest quarterly based on the outstanding balance."""
        loc_data = self.data.copy()
        loc_data['Loan Balance'] = 0.0
        loc_data['Interest Paid'] = 0.0 # Tracks interest drawn as contributions
        num_periods = len(loc_data)

        delay = int(delay_quarters)
        if delay <= 0:
            print("LOC Delay is zero or negative. No LOC applied.")
            return loc_data, pd.Series([0.0]*num_periods), pd.Series([0.0]*num_periods)

        print(f"Applying LOC: Delaying calls by {delay} quarters. Interest at {interest_rate*100:.2f}% drawn quarterly on outstanding balance.")
        quarterly_rate = interest_rate / 4.0

        # Store details of calls to be delayed: (original_period, amount, repayment_period)
        delayed_calls = []
        for i in range(num_periods):
            contrib = self.data.loc[i, 'Contribution']
            if contrib > 0:
                repayment_period = min(i + delay, num_periods - 1)
                delayed_calls.append({'original': i, 'amount': contrib, 'repayment': repayment_period})
                # Remove contribution from original position in the modified data
                loc_data.loc[i, 'Contribution'] = 0
                print(f"LOC: Delaying ${contrib:,.2f} from period {i} to period {repayment_period}")

        current_loc_balance = 0.0
        # Iterate through each period to calculate balance and interest
        for p in range(num_periods):
            balance_at_start_of_period = current_loc_balance

            # Add newly delayed contributions to the balance *after* the original period
            for call in delayed_calls:
                if call['original'] == p:
                    current_loc_balance += call['amount']

            # Calculate and draw interest based on the balance at the start of the period
            if balance_at_start_of_period > 0:
                interest_draw = balance_at_start_of_period * quarterly_rate
                # Add interest draw as a contribution in the current period
                loc_data.loc[p, 'Contribution'] += interest_draw
                loc_data.loc[p, 'Interest Paid'] += interest_draw # Track interest cost
                print(f"  Period {p}: Interest Draw ${interest_draw:,.2f} on balance ${balance_at_start_of_period:,.2f}")


            # Process principal repayments scheduled for this period
            principal_repaid_this_period = 0
            for call in delayed_calls:
                if call['repayment'] == p:
                    # Add the principal back as a contribution
                    loc_data.loc[p, 'Contribution'] += call['amount']
                    # Reduce the outstanding balance
                    current_loc_balance -= call['amount']
                    principal_repaid_this_period += call['amount']

            if principal_repaid_this_period > 0:
                 print(f"  Period {p}: Principal Repayment (Contribution) ${principal_repaid_this_period:,.2f}")


            # Store the end-of-period loan balance
            loc_data.loc[p, 'Loan Balance'] = current_loc_balance
            # Clamp balance at 0 if minor floating point issues occur
            if current_loc_balance < 1e-6: # Using a small threshold instead of == 0
                current_loc_balance = 0.0


        total_interest_drawn = loc_data['Interest Paid'].sum()
        print(f"LOC: Total interest drawn via contributions: ${total_interest_drawn:,.2f}")

        # Return modified data, the calculated loan balance series, and the interest drawn series
        return loc_data, loc_data['Loan Balance'], loc_data['Interest Paid']
    
    def apply_nav_loan(self, interest_rate, size_percentage, term_years, origination_year):
        """Model a NAV loan impact on fund returns with quarterly interest payments."""
        modified_data = self.data.copy()
        modified_data['Loan Balance'] = 0.0
        modified_data['Interest Paid'] = 0.0
        num_periods = len(modified_data)

        # --- Calculate Loan Origination ---
        # Year N ends *after* N*4 quarters. Index is N*4.
        # Ensure index is within bounds (0 to num_periods-1)
        loan_index = min(max(0, origination_year * 4), num_periods - 1)

        # Ensure loan index is valid and there's NAV to borrow against
        if loan_index < 0 or modified_data.loc[loan_index, 'NAV'] <= 0:
             print(f"Warning: Cannot originate NAV loan at end of Year {origination_year}. NAV is non-positive or index invalid. No loan applied.")
             # Return unmodified data and zero series for loans/interest
             return modified_data, pd.Series([0.0]*num_periods), pd.Series([0.0]*num_periods)

        # Calculate loan amount based on NAV at year-end
        nav_at_loan = modified_data.loc[loan_index, 'NAV']
        loan_amount = nav_at_loan * size_percentage / 100

        if loan_amount <= 0:
            print(f"Warning: Calculated loan amount is zero or negative at end of Year {origination_year}. No loan applied.")
            return modified_data, pd.Series([0.0]*num_periods), pd.Series([0.0]*num_periods)

        print(f"NAV Loan: ${loan_amount:,.2f} taken at period {loan_index} (End of Year {origination_year})")

        # --- Loan Disbursement ---
        # Distribute loan proceeds immediately (increases distribution in the period *after* origination)
        disbursement_index = min(loan_index + 1, num_periods - 1)
        modified_data.loc[disbursement_index, 'Distribution'] += loan_amount
        print(f"NAV Loan: Disbursing ${loan_amount:,.2f} at period {disbursement_index}")

        # --- Interest Payments & Loan Balance Tracking ---
        interest_rate_quarterly = interest_rate / 4
        quarters_in_term = int(term_years * 4)
        current_loan_balance = loan_amount

        # Track loan balance from disbursement period onwards
        modified_data.loc[disbursement_index:, 'Loan Balance'] = loan_amount

        # Calculate and deduct interest quarterly until maturity
        for i in range(1, quarters_in_term + 1):
            interest_payment_period = min(disbursement_index + i, num_periods - 1)

            # Stop if we go past the data range
            if interest_payment_period >= num_periods:
                print(f"Warning: Loan term extends beyond data range. Interest calculation stopped at period {num_periods-1}.")
                break

            # Calculate interest for the quarter based on balance at start (which is current_loan_balance)
            interest_payment = current_loan_balance * interest_rate_quarterly

            # Deduct interest payment from distributions for that quarter
            modified_data.loc[interest_payment_period, 'Distribution'] -= interest_payment
            modified_data.loc[interest_payment_period, 'Interest Paid'] += interest_payment # Track interest paid

            print(f"NAV Loan: Interest payment of ${interest_payment:,.2f} at period {interest_payment_period}")

            # Update loan balance if needed (assuming interest-only for now)
            # If it were amortizing, principal would be paid here too.

        # --- Loan Repayment ---
        maturity_index = min(disbursement_index + quarters_in_term, num_periods - 1)

        # Ensure maturity index is valid
        if maturity_index >= disbursement_index:
             # Subtract principal repayment at maturity
             modified_data.loc[maturity_index, 'Distribution'] -= loan_amount
             # Set loan balance to zero after repayment
             if maturity_index + 1 < num_periods:
                 modified_data.loc[maturity_index + 1:, 'Loan Balance'] = 0.0
             # Ensure the last period's balance is zero if maturity is the last period
             modified_data.loc[maturity_index, 'Loan Balance'] = 0.0


             print(f"NAV Loan: Repaying principal ${loan_amount:,.2f} at period {maturity_index}")
             total_interest_paid = modified_data['Interest Paid'].sum()
             print(f"NAV Loan: Total interest paid: ${total_interest_paid:,.2f}")
        else:
            print(f"Warning: Loan maturity index ({maturity_index}) is before disbursement index ({disbursement_index}). Repayment not applied.")


        return modified_data, modified_data['Loan Balance'], modified_data['Interest Paid']

    def apply_combined(self, loc_interest_rate, loc_delay_quarters,
                       nav_interest_rate, nav_size_percentage, nav_term_years, nav_origination_year):
        """Apply both LOC and NAV loan strategies together.

        First applies the LOC strategy, then applies the NAV loan on top of that.
        Combines loan balances and interest costs.
        """
        # First apply LOC (using the updated logic with balance tracking)
        loc_data, loc_loans, loc_interest = self.apply_loc(loc_interest_rate, loc_delay_quarters)

        # Create a temporary fund object to hold the LOC-modified data
        temp_fund = PEFund.__new__(PEFund)
        temp_fund.data = loc_data # This data now includes LOC contributions, interest draws, and loan balance

        # Then apply NAV loan on top of the LOC data
        combined_data, combined_nav_loans, combined_nav_interest = temp_fund.apply_nav_loan(
            nav_interest_rate, nav_size_percentage, nav_term_years, nav_origination_year
        )

        # Combine loan/interest effects
        # The combined_data already reflects cash flows from both LOC (contribs) and NAV (distribs)
        # We need to combine the tracked balances and interest costs for reporting

        # Combine Loan Balances: Add the LOC balance and NAV loan balance for each period
        # Note: loc_loans is from the *initial* LOC application, combined_nav_loans is from the NAV loan applied *after* LOC.
        # We need the LOC loan balance from the *final* combined_data if it were tracked through apply_nav_loan,
        # but apply_nav_loan overwrites 'Loan Balance'.
        # For simplicity here, we'll add the initially calculated LOC balance to the final NAV balance.
        # A more rigorous approach might track them separately throughout.
        final_loans = loc_loans + combined_nav_loans # Sum balances from both sources

        # Combine Interest Costs: Sum the interest drawn via LOC and interest paid via NAV
        final_interest = loc_interest + combined_nav_interest

        # Ensure the combined_data DataFrame has the final *combined* loan/interest columns for display
        combined_data['Loan Balance'] = final_loans
        combined_data['Interest Paid'] = final_interest # This represents total interest cost

        return combined_data, final_loans, final_interest 