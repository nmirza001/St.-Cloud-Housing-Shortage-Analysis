# ==============================================================================
# St. Cloud Housing Shortage Analysis - Economics Research Version
# ==============================================================================
# Following Brookings Institute methodology exactly
# Full data logging and raw data preservation for research transparency
# ==============================================================================

import pandas as pd
import numpy as np
import requests
import json
import time
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Set up comprehensive logging
def setup_logging():
    """Set up detailed logging for research transparency"""
    log_filename = f"housing_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("ST. CLOUD HOUSING SHORTAGE ANALYSIS - RESEARCH LOG")
    logger.info("="*80)
    logger.info("Methodology: NAHB Vacancy Rate Method (Brookings Institute)")
    logger.info("Research Question: Impact of Sherburne County inclusion on shortage estimates")
    logger.info("="*80)
    return logger

# Census ACS Housing Variables (exact B-tables from Brookings methodology)
HOUSING_VARIABLES = {
    # Core occupancy data
    "B25001_001E": "Total housing units",
    "B25003_001E": "Total occupied housing units", 
    "B25003_002E": "Owner-occupied housing units",
    "B25003_003E": "Renter-occupied housing units",
    
    # Vacancy by type (B25004 table - critical for NAHB method)
    "B25004_001E": "Total vacant housing units",
    "B25004_002E": "For rent",
    "B25004_003E": "Rented, not occupied", 
    "B25004_004E": "For sale only",
    "B25004_005E": "Sold, not occupied",
    "B25004_006E": "For seasonal, recreational, or occasional use",
    "B25004_007E": "For migrant workers",
    "B25004_008E": "Other vacant",
    
    # Additional context
    "B25002_002E": "Occupied housing units (verification)",
    "B25002_003E": "Vacant housing units (verification)",
}

# Geographic definitions following OMB MSA standards
RESEARCH_GEOGRAPHIES = {
    "St_Cloud_MSA_Official": {
        "description": "St. Cloud MSA (Stearns & Benton Counties) - Official OMB Definition",
        "for": "county:009,145",  # Benton (009), Stearns (145)
        "in": "state:27",         # Minnesota
        "component_counties": ["Benton", "Stearns"]
    },
    "Stearns_County": {
        "description": "Stearns County - Primary MSA component",
        "for": "county:145",
        "in": "state:27",
        "component_counties": ["Stearns"]
    },
    "Benton_County": {
        "description": "Benton County - Secondary MSA component", 
        "for": "county:009",
        "in": "state:27",
        "component_counties": ["Benton"]
    },
    "Sherburne_County": {
        "description": "Sherburne County - Adjacent to MSA",
        "for": "county:141",
        "in": "state:27", 
        "component_counties": ["Sherburne"]
    },
    "St_Cloud_City_All": {
        "description": "St. Cloud City - All county portions",
        "for": "place:56896",  # St. Cloud city FIPS
        "in": "state:27",
        "component_counties": ["Stearns", "Benton", "Sherburne"]
    }
}

class HousingDataLogger:
    """Handles all data collection and logging for research transparency"""
    
    def __init__(self, api_key: str, logger: logging.Logger):
        self.api_key = api_key
        self.logger = logger
        self.base_url = "https://api.census.gov/data"
        self.raw_data_cache = {}
        self.api_call_log = []
        
        # Create directories for data storage
        os.makedirs("raw_data", exist_ok=True)
        os.makedirs("processed_data", exist_ok=True)
        os.makedirs("methodology_logs", exist_ok=True)
        
        self.logger.info(f"Data collection initialized with API key: {api_key[:8]}...")
        self.logger.info("Directory structure created for data preservation")

    def make_census_api_call(self, year: int, geography: str, geo_params: Dict) -> Optional[pd.DataFrame]:
        """Make Census API call with full logging and error handling"""
        
        # Construct the API request
        variables = list(HOUSING_VARIABLES.keys())
        var_string = ",".join(variables)
        url = f"{self.base_url}/{year}/acs/acs5"
        
        # Extract only the Census API parameters (exclude our metadata)
        census_params = {
            'get': var_string,
            'key': self.api_key,
            'for': geo_params['for'],
            'in': geo_params['in']
        }
        
        # Log the API request (include metadata for research documentation)
        request_info = {
            'timestamp': datetime.now().isoformat(),
            'year': year,
            'geography': geography,
            'description': geo_params.get('description', ''),
            'url': url,
            'census_parameters': census_params.copy()  # Copy to avoid API key in logs
        }
        request_info['census_parameters']['key'] = f"{self.api_key[:8]}..."  # Mask API key
        
        self.logger.info(f"API Request: {year} data for {geography}")
        self.logger.info(f"Description: {geo_params.get('description', '')}")
        self.logger.info(f"URL: {url}")
        self.logger.info(f"Census Parameters: {request_info['census_parameters']}")
        
        # Attempt the API call with retries
        for attempt in range(3):
            try:
                response = requests.get(url, params=census_params, timeout=30)
                response.raise_for_status()
                
                # Log successful response
                self.logger.info(f"API Response: Status {response.status_code}, Size: {len(response.content)} bytes")
                
                # Parse JSON response
                data = response.json()
                request_info['response_size'] = len(data)
                request_info['success'] = True
                
                if len(data) <= 1:
                    self.logger.warning(f"Empty data returned for {geography} {year}")
                    request_info['empty_response'] = True
                    self.api_call_log.append(request_info)
                    return None
                
                # Convert to DataFrame
                df = pd.DataFrame(data[1:], columns=data[0])
                
                # Save raw data
                raw_filename = f"raw_data/{geography}_{year}_raw.json"
                with open(raw_filename, 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Convert numeric columns and log any conversion issues
                conversion_log = {}
                for col in variables:
                    if col in df.columns:
                        original_values = df[col].copy()
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                        
                        # Log conversion issues
                        null_count = original_values.isnull().sum()
                        if null_count > 0:
                            conversion_log[col] = f"{null_count} null values converted to 0"
                
                if conversion_log:
                    self.logger.warning(f"Data conversion issues: {conversion_log}")
                    request_info['conversion_issues'] = conversion_log
                
                # Save processed data
                processed_filename = f"processed_data/{geography}_{year}_processed.csv"
                df.to_csv(processed_filename, index=False)
                
                self.logger.info(f"Data saved: {raw_filename}, {processed_filename}")
                self.logger.info(f"Data shape: {df.shape}, Variables: {len(variables)}")
                
                self.api_call_log.append(request_info)
                return df
                
            except requests.exceptions.HTTPError as e:
                self.logger.error(f"HTTP Error (attempt {attempt+1}/3): {e}")
                request_info['error'] = str(e)
                if attempt == 2:  # Last attempt
                    request_info['success'] = False
                    self.api_call_log.append(request_info)
                    return None
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                request_info['error'] = str(e)
                request_info['success'] = False
                self.api_call_log.append(request_info)
                return None
        
        return None

    def save_api_log(self):
        """Save complete API call log for research documentation"""
        log_filename = f"methodology_logs/api_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(log_filename, 'w') as f:
            json.dump(self.api_call_log, f, indent=2)
        self.logger.info(f"API call log saved: {log_filename}")

class BrookingsVacancyAnalyzer:
    """Implements the exact NAHB vacancy rate methodology from Brookings"""
    
    def __init__(self, data_logger: HousingDataLogger, logger: logging.Logger):
        self.data_logger = data_logger
        self.logger = logger
        self.results = {}
        
        self.logger.info("Vacancy Rate Analyzer initialized")
        self.logger.info("Methodology: NAHB approach as described in Brookings Institute report")

    def calculate_nahb_vacancy_metrics(self, df: pd.DataFrame, geography: str, year: int) -> Dict:
        """Calculate vacancy metrics following exact NAHB methodology"""
        
        self.logger.info(f"Calculating NAHB vacancy metrics for {geography} ({year})")
        
        # Step 1: Extract basic occupancy data
        total_units = df["B25001_001E"].sum()
        occupied_units = df["B25003_002E"].sum() + df["B25003_003E"].sum()  # Owner + Renter occupied
        owner_occupied = df["B25003_002E"].sum()
        renter_occupied = df["B25003_003E"].sum()
        
        # Step 2: Extract vacancy by detailed type (critical for NAHB)
        vacant_for_rent = df["B25004_002E"].sum()
        vacant_rented_not_occupied = df["B25004_003E"].sum()
        vacant_for_sale = df["B25004_004E"].sum()
        vacant_sold_not_occupied = df["B25004_005E"].sum()
        
        # Step 3: Identify units excluded from NAHB shortage calculation
        vacant_seasonal = df["B25004_006E"].sum()  # Seasonal/recreational
        vacant_migrant = df["B25004_007E"].sum()   # Migrant workers
        vacant_other = df["B25004_008E"].sum()     # Other vacant
        
        # Step 4: Calculate market-available vacant units (NAHB method)
        market_vacant_rental = vacant_for_rent + vacant_rented_not_occupied
        market_vacant_owner = vacant_for_sale + vacant_sold_not_occupied
        excluded_vacant = vacant_seasonal + vacant_migrant + vacant_other
        
        # Step 5: Calculate housing stock by tenure (market-relevant only)
        owner_housing_stock = owner_occupied + market_vacant_owner
        rental_housing_stock = renter_occupied + market_vacant_rental
        total_market_stock = owner_housing_stock + rental_housing_stock
        
        # Step 6: Calculate vacancy rates (NAHB definition)
        owner_vacancy_rate = market_vacant_owner / owner_housing_stock if owner_housing_stock > 0 else 0
        rental_vacancy_rate = market_vacant_rental / rental_housing_stock if rental_housing_stock > 0 else 0
        
        # Log detailed breakdown
        self.logger.info(f"Housing stock breakdown for {geography}:")
        self.logger.info(f"  Total units: {total_units:,}")
        self.logger.info(f"  Occupied: {occupied_units:,} (Owner: {owner_occupied:,}, Renter: {renter_occupied:,})")
        self.logger.info(f"  Market vacant: {market_vacant_owner + market_vacant_rental:,} (Owner: {market_vacant_owner:,}, Rental: {market_vacant_rental:,})")
        self.logger.info(f"  Excluded vacant: {excluded_vacant:,} (Seasonal: {vacant_seasonal:,}, Migrant: {vacant_migrant:,}, Other: {vacant_other:,})")
        self.logger.info(f"  Vacancy rates: Owner {owner_vacancy_rate:.3f}, Rental {rental_vacancy_rate:.3f}")
        
        # Data validation checks
        if owner_vacancy_rate > 0.15:
            self.logger.warning(f"Owner vacancy rate ({owner_vacancy_rate:.1%}) unusually high for {geography}")
        if rental_vacancy_rate > 0.25:
            self.logger.warning(f"Rental vacancy rate ({rental_vacancy_rate:.1%}) unusually high for {geography}")
        if owner_vacancy_rate < 0.005:
            self.logger.warning(f"Owner vacancy rate ({owner_vacancy_rate:.1%}) unusually low for {geography}")
        
        metrics = {
            'total_units': int(total_units),
            'occupied_units': int(occupied_units),
            'owner_occupied': int(owner_occupied),
            'renter_occupied': int(renter_occupied),
            'owner_housing_stock': int(owner_housing_stock),
            'rental_housing_stock': int(rental_housing_stock), 
            'total_market_stock': int(total_market_stock),
            'market_vacant_owner': int(market_vacant_owner),
            'market_vacant_rental': int(market_vacant_rental),
            'excluded_vacant': int(excluded_vacant),
            'owner_vacancy_rate': owner_vacancy_rate,
            'rental_vacancy_rate': rental_vacancy_rate,
            'geography': geography,
            'year': year
        }
        
        # Save detailed metrics
        metrics_filename = f"processed_data/{geography}_{year}_nahb_metrics.json"
        with open(metrics_filename, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        return metrics

    def calculate_historical_normal_rates(self, geography: str, geo_params: Dict) -> Dict:
        """Calculate historical normal vacancy rates using multiple baseline approaches"""
        
        self.logger.info(f"Calculating historical normal rates for {geography}")
        
        # Define different baseline approaches from Brookings article
        baseline_approaches = {
            'NAHB_Official': {
                'years': list(range(2009, 2022)),  # 2009-2021 (NAHB includes pandemic!)
                'description': 'NAHB method: 2009-2021 including pandemic (Brookings Article)'
            },
            'Zandi_Style': {
                'years': list(range(2013, 2020)),  # 2013-2019 (stable growth period)
                'description': 'Zandi-style: Stable growth period excluding crises'
            },
            'Academic_Conservative': {
                'years': list(range(2015, 2020)),  # 2015-2019 (most recent stable)
                'description': 'Academic approach: Most recent stable period only'
            }
        }
        
        all_results = {}
        
        for approach_name, approach_config in baseline_approaches.items():
            self.logger.info(f"Testing {approach_name}: {approach_config['description']}")
            baseline_years = approach_config['years']
            self.logger.info(f"Baseline years: {baseline_years}")
            
            historical_owner_rates = []
            historical_rental_rates = []
            valid_years = []
            
            for year in baseline_years:
                self.logger.info(f"Processing {approach_name} data for {year}")
                
                df = self.data_logger.make_census_api_call(year, geography, geo_params)
                if df is not None and len(df) > 0:
                    metrics = self.calculate_nahb_vacancy_metrics(df, geography, year)
                    
                    # Quality control: only use years with reasonable data
                    if (metrics['owner_housing_stock'] > 50 and 
                        metrics['rental_housing_stock'] > 25 and
                        0.003 <= metrics['owner_vacancy_rate'] <= 0.12 and
                        0.02 <= metrics['rental_vacancy_rate'] <= 0.20):
                        
                        historical_owner_rates.append(metrics['owner_vacancy_rate'])
                        historical_rental_rates.append(metrics['rental_vacancy_rate'])
                        valid_years.append(year)
                        self.logger.info(f"  {year}: Valid data (Owner: {metrics['owner_vacancy_rate']:.3f}, Rental: {metrics['rental_vacancy_rate']:.3f})")
                    else:
                        self.logger.warning(f"  {year}: Data quality issues, excluding from baseline")
                else:
                    self.logger.warning(f"  {year}: No data available")
            
            # Calculate normal rates for this approach
            if len(historical_owner_rates) >= 3:
                if len(historical_owner_rates) >= 5:
                    # Use median for robustness
                    normal_owner_rate = np.median(historical_owner_rates)
                    normal_rental_rate = np.median(historical_rental_rates)
                    method = 'median'
                else:
                    normal_owner_rate = np.mean(historical_owner_rates)
                    normal_rental_rate = np.mean(historical_rental_rates)
                    method = 'mean'
                
                # Calculate standard deviation to show volatility
                owner_std = np.std(historical_owner_rates) if len(historical_owner_rates) > 1 else 0
                rental_std = np.std(historical_rental_rates) if len(historical_rental_rates) > 1 else 0
                
                all_results[approach_name] = {
                    'normal_owner_rate': normal_owner_rate,
                    'normal_rental_rate': normal_rental_rate,
                    'baseline_years': valid_years,
                    'data_points': len(historical_owner_rates),
                    'owner_rate_series': historical_owner_rates,
                    'rental_rate_series': historical_rental_rates,
                    'owner_std_dev': owner_std,
                    'rental_std_dev': rental_std,
                    'calculation_method': method,
                    'description': approach_config['description']
                }
                
                self.logger.info(f"{approach_name} results: Owner {normal_owner_rate:.3f} (±{owner_std:.3f}), Rental {normal_rental_rate:.3f} (±{rental_std:.3f})")
            else:
                self.logger.warning(f"{approach_name}: Insufficient data points")
        
        # Choose primary method (NAHB official for main analysis)
        if 'NAHB_Official' in all_results:
            primary_method = 'NAHB_Official'
        elif 'Zandi_Style' in all_results:
            primary_method = 'Zandi_Style'
        elif 'Academic_Conservative' in all_results:
            primary_method = 'Academic_Conservative'
        else:
            # Fallback to defaults
            self.logger.warning("No baseline methods succeeded, using regional defaults")
            all_results['Default'] = {
                'normal_owner_rate': 0.018,
                'normal_rental_rate': 0.075,
                'baseline_years': [],
                'data_points': 0,
                'calculation_method': 'default',
                'description': 'Regional defaults due to data unavailability'
            }
            primary_method = 'Default'
        
        # Log comparison of methods
        self.logger.info("="*60)
        self.logger.info("BASELINE METHOD COMPARISON:")
        for method_name, results in all_results.items():
            owner_rate = results['normal_owner_rate']
            rental_rate = results['normal_rental_rate']
            data_points = results['data_points']
            self.logger.info(f"{method_name}: Owner {owner_rate:.3f}, Rental {rental_rate:.3f} ({data_points} years)")
        self.logger.info("="*60)
        
        # Return primary method with comparison data
        final_results = all_results[primary_method].copy()
        final_results['method_comparison'] = all_results
        final_results['primary_method'] = primary_method
        
        # Save detailed baseline analysis
        baseline_filename = f"methodology_logs/{geography}_baseline_comparison.json"
        with open(baseline_filename, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        self.logger.info(f"Using {primary_method} as primary method")
        self.logger.info(f"Baseline comparison saved: {baseline_filename}")
        
        return final_results

    def calculate_nahb_shortage(self, current_metrics: Dict, normal_rates: Dict) -> Dict:
        """Calculate housing shortage using exact NAHB methodology"""
        
        geography = current_metrics['geography']
        self.logger.info(f"Calculating NAHB shortage for {geography}")
        
        # NAHB Method: Calculate target vacant units needed for normal vacancy rates
        target_owner_vacant = current_metrics['owner_housing_stock'] * normal_rates['normal_owner_rate']
        target_rental_vacant = current_metrics['rental_housing_stock'] * normal_rates['normal_rental_rate']
        
        # Shortage = additional vacant units needed to reach normal vacancy
        owner_shortage = max(0, target_owner_vacant - current_metrics['market_vacant_owner'])
        rental_shortage = max(0, target_rental_vacant - current_metrics['market_vacant_rental'])
        total_shortage = owner_shortage + rental_shortage
        
        # Calculate shortage as percentage of stock
        owner_shortage_pct = owner_shortage / current_metrics['owner_housing_stock'] if current_metrics['owner_housing_stock'] > 0 else 0
        rental_shortage_pct = rental_shortage / current_metrics['rental_housing_stock'] if current_metrics['rental_housing_stock'] > 0 else 0
        total_shortage_pct = total_shortage / current_metrics['total_market_stock'] if current_metrics['total_market_stock'] > 0 else 0
        
        shortage_analysis = {
            'geography': geography,
            'target_owner_vacant': round(target_owner_vacant),
            'target_rental_vacant': round(target_rental_vacant),
            'current_owner_vacant': current_metrics['market_vacant_owner'],
            'current_rental_vacant': current_metrics['market_vacant_rental'],
            'owner_shortage': round(owner_shortage),
            'rental_shortage': round(rental_shortage), 
            'total_shortage': round(total_shortage),
            'owner_shortage_pct': owner_shortage_pct,
            'rental_shortage_pct': rental_shortage_pct,
            'total_shortage_pct': total_shortage_pct,
            'methodology': 'NAHB_vacancy_rate_method'
        }
        
        self.logger.info(f"NAHB Shortage Results for {geography}:")
        self.logger.info(f"  Target vacant units: Owner {target_owner_vacant:.0f}, Rental {target_rental_vacant:.0f}")
        self.logger.info(f"  Current vacant units: Owner {current_metrics['market_vacant_owner']}, Rental {current_metrics['market_vacant_rental']}")
        self.logger.info(f"  Shortage: Owner {owner_shortage:.0f}, Rental {rental_shortage:.0f}, Total {total_shortage:.0f}")
        self.logger.info(f"  Shortage as % of stock: {total_shortage_pct:.1%}")
        
        return shortage_analysis

    def analyze_geography(self, geography: str, geo_config: Dict, current_year: int = 2022):
        """Complete NAHB analysis for one geography"""
        
        self.logger.info("="*80)
        self.logger.info(f"BEGINNING ANALYSIS: {geography}")
        self.logger.info(f"Description: {geo_config['description']}")
        self.logger.info("="*80)
        
        # Step 1: Get current year data
        self.logger.info(f"Step 1: Collecting {current_year} housing data")
        current_df = self.data_logger.make_census_api_call(current_year, geography, geo_config)
        
        if current_df is None or current_df.empty:
            self.logger.error(f"No current data available for {geography}")
            return
        
        # Step 2: Calculate current vacancy metrics
        self.logger.info(f"Step 2: Calculating current vacancy metrics")
        current_metrics = self.calculate_nahb_vacancy_metrics(current_df, geography, current_year)
        
        # Step 3: Calculate historical normal rates
        self.logger.info(f"Step 3: Calculating historical normal vacancy rates")
        normal_rates = self.calculate_historical_normal_rates(geography, geo_config)
        
        # Step 4: Calculate shortage using NAHB method
        self.logger.info(f"Step 4: Calculating housing shortage")
        shortage_analysis = self.calculate_nahb_shortage(current_metrics, normal_rates)
        
        # Step 5: Combine all results
        complete_results = {
            **current_metrics,
            **normal_rates,
            **shortage_analysis,
            'analysis_date': datetime.now().isoformat(),
            'methodology': 'NAHB_Brookings_vacancy_rate_method'
        }
        
        self.results[geography] = complete_results
        
        # Save complete analysis
        results_filename = f"processed_data/{geography}_complete_analysis.json"
        with open(results_filename, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
        
        self.logger.info(f"Analysis complete for {geography}")
        self.logger.info(f"Results saved: {results_filename}")

def main():
    """Main research execution following economics methodology"""
    
    # Set up research environment
    logger = setup_logging()
    logger.info("Research execution started")
    
    # Get API key
    API_KEY = os.getenv('CENSUS_API_KEY', 'b383b2f1edc06d405781cc24a7250c56ccf208c2')
    logger.info("Census API key loaded")
    
    # Initialize research components
    try:
        data_logger = HousingDataLogger(API_KEY, logger)
        analyzer = BrookingsVacancyAnalyzer(data_logger, logger)
    except Exception as e:
        logger.error(f"Failed to initialize research components: {e}")
        return
    
    # Execute analysis for each geography
    logger.info("Beginning geographic analysis sequence")
    
    for geography, geo_config in RESEARCH_GEOGRAPHIES.items():
        try:
            analyzer.analyze_geography(geography, geo_config)
        except Exception as e:
            logger.error(f"Analysis failed for {geography}: {e}")
            continue
    
    # Save comprehensive logs
    data_logger.save_api_log()
    
    # Create final research summary
    if analyzer.results:
        summary_df = pd.DataFrame.from_dict(analyzer.results, orient='index')
        summary_filename = f"processed_data/research_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_filename)
        
        logger.info("="*80)
        logger.info("RESEARCH SUMMARY")
        logger.info("="*80)
        
        for geography, results in analyzer.results.items():
            logger.info(f"{geography}: {results['total_shortage']:,} unit shortage ({results['total_shortage_pct']:.1%})")
        
        logger.info(f"Complete research summary saved: {summary_filename}")
        logger.info(f"All raw data preserved in: raw_data/")
        logger.info(f"All processed data saved in: processed_data/")
        logger.info(f"Methodology logs in: methodology_logs/")
    
    logger.info("="*80)
    logger.info("RESEARCH EXECUTION COMPLETE")
    logger.info("="*80)

if __name__ == "__main__":
    main()