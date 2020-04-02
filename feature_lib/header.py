import os
pj = os.path.join
feature_columns = {
    'opportunity':{
        'dummy':[
            'AccountId', 
            'Building_uuid__c', 
            'StageName', 
            'LastModifiedDate', 
            'Total_Contract_Revenue__c',
        ]
    },
    'account':{
        'dummy':[
            'Id',
            'Cleansed_Unomy_Company_Name__c',
            'Unomy_Company_ID_Formula__c',
        ],
        'continuous':[
            'Account_FTE__c',
            'Number_Of_Open_Opportunities__c',
            'Number_of_Full_Time_Employees__c',
            'Number_of_Opportunity_Reservables__c',
            'Unomy_Company_Size_Formula__c',
            'Won_Opportunities__c',
            'Unomy_Estimated_Revenue_Formula__c',
            'Unomy_Facebook_Likes__c',
            'Unomy_Twitter_Followers__c',
            'Unomy_Alexa_Global_Rank__c',
            'Unomy_Alexa_US_Rank__c',
        ],
        'discrete':[
            'Account_Type__c',
            'Account_Market__c',
            'Country_Code__c',
            'CurrencyIsoCode',
            'Member_Qualification__c',
            'Org_Identification_Status__c',
            'ROE_Segment__c',
            'Segment_Detail__c',
            'Total_Desks_Sold__c',
            'Unomy_Industry__c',
            'Update_HQ_Market_Status__c',
            'ID_Status2__c',
        ],
        # 'Interested_in_Number_of_Desks__c',
        #'Company_Trend__c',
    },
    'company':{
        'dummy': [
            'CI_Company_ID__c',
            'LastModifiedDate',
            'Company_Name__c', 
        ],
        'continuous': [
            'Size__c',
            'Revenue__c'
        ],
        'discrete': [
            'Industry__c',
        ], 
        #'Company_Trend__c',
    },
    'building':{
        'dummy': [
            "Address__c",
            "Building_ID__c",
            "Country__c",
            "Id",
            "Name",
            "Sort_Order__c",
            "UUID__c",
            "SystemModstamp", 
            "City__c",
            "State__c",
            "Geography__c",
            "Location_No_City__c",
            "Cluster_Name__c",
            
        ],
        'continuous': [
            "Available_Desks__c",
            "Available_Offices__c",
            "Desks_Occupied_Right_Now__c",
            "Desks_Ready_for_Occupany__c",
            "Occupancy_Rate__c",
            "Total_Desks__c",
            "Total_Offices__c",
            "Unavailable_Desks__c",
            "Unavailable_Offices__c",
        ],
        'discrete':[
            "Cluster__c",
            "Contract_Status__c",
            "CurrencyIsoCode",
            "Gate__c",
            "HQ_by_WeWork__c",
            "Market__c",
            "Maximum_Tour_Days__c",
            "NOT_Tourable_Sellable__c",
            "Occupancy_Rating__c",
            "Open_Studio__c",
            "Portfolio_Standard_Name__c",
            "Simultaneous_Tours__c",
            "Territory_Name__c",
            "Time_Zone__c",
            "Tour_Spacing__c",
            "WeWork_Labs__c",
        ],  
        #"Building_to_Gate_A__c",
        #"OwnerId", single value
        #"Portfolio_Name__c",
        #"Postal_Code__c",
    },
    'geography':{
        'dummy': [
            'Country_Code__c',
            'Country__c',
            'Geocode__Latitude__s',
            'Geocode__Longitude__s',
            'Id',
            'Location_Type__c',
            'Formatted_Address__c',
            'Nearest_Building__c',
            'Previous_Building__c',
            'City__c',
            'State__c',
            'Place_ID__c',
            'Zip_Postal_Code__c'
        ]
    },

    'location_scorecard':{
        'dummy':[
            'Unnamed: 0',
            'atlas_location_uuid',
            'longitude',
            'latitude',
            'city',
        ],
        'continuous':[
            'score_predicted_eo', 'score_employer', 'num_emp_weworkcore', 'num_poi_weworkcore',
            'pct_wwcore_employee', 'pct_wwcore_business', 'num_retail_stores', 'num_doctor_offices',
            'num_eating_places', 'num_drinking_places', 'num_hotels', 'num_fitness_gyms',
            'population_density', 'pct_female_population', 'median_age', 'income_per_capita',
            'pct_masters_degree', 'walk_score', 'bike_score' 
        ]
    }
}

feature_mappings = {
    'opportunity': {
        'AccountId': 'account_id', 
        'Building_uuid__c': 'atlas_location_uuid', 
        'LastModifiedDate': 'timestamp'
    },
    'account': {
        'Id': 'account_id',
        'Unomy_Company_ID_Formula__c': 'company_id',
        'Unomy_Estimated_Revenue_Formula__c': 'unomy_revenue',
        'Unomy_Company_Size_Formula__c': 'unomy_company_size',
        'Unomy_Industry__c': 'unomy_industry',
        # 'Industry': 'industry',
        'Account_FTE__c': 'headcount'
    },
    'building': {
        'UUID__c': 'atlas_location_uuid', 
        'Country__c': 'country',
        'Geography__c': 'geo_id',
        'City__c': 'city'
    },
    'geography': {
        'Geocode__Latitude__s': 'lat',
        'Geocode__Longitude__s': 'long',
        'Id': 'geo_id',
    },
    'company': {
        'CI_Company_ID__c': 'company_id',
        'Industry__c': 'industry',
        'Size__c': 'company_size',
        'Revenue__c': 'revenue'
    }

}

DATAPATH = '/Users/kanghuang/Documents/work/location_recommendation/salesforce_data'
CACHEPATH = '/Users/kanghuang/Documents/work/location_recommendation/salesforce_model/cache'
OUTPUTPATH = '/Users/kanghuang/Documents/work/location_recommendation/salesforce_model/artifacts'
datapaths= {
    'opportunity': pj(DATAPATH, 'sfdc_opportunities_all.csv'),
    'account': pj(DATAPATH, 'sfdc_accounts_all.csv'),
    'building': pj(DATAPATH, 'sfdc_buildings_all.csv'),
    'geography': pj(DATAPATH, 'sfdc_geography_all.csv'),
    'company': pj(DATAPATH, 'sfdc_company_all.csv'),
    'tour': pj(DATAPATH, 'sfdc_tour_all.csv'),
    'location_scorecard': pj(DATAPATH, 'location_scorecard_200106.csv'),
}


feature_types = {
    'salesforce':{
        'not_feat_col':[
            'Unnamed: 0', 
            'account_id',
            'Cleansed_Unomy_Company_Name__c',
        ],

        'dummy_col_name':[
            'Account_Market__c',
            'Account_Type__c',
            'Country_Code__c',
            'CurrencyIsoCode',
            'Member_Qualification__c',
            'ROE_Segment__c',
            'Segment_Detail__c',
            'Update_HQ_Market_Status__c',
            'Unomy_Industry__c',
            'ID_Status2__c',
            'Org_Identification_Status__c'
        ],
        'cont_col_name':{
            'norm':[
                'Unomy_Estimated_Revenue_Formula__c',
                'Unomy_Company_Size_Formula__c',
                'Account_FTE__c',
                'Unomy_Facebook_Likes__c',
                'Unomy_Twitter_Followers__c',
            ],
            'minmax':[
                'Interested_in_Number_of_Desks__c',
                'Number_Of_Open_Opportunities__c',
                'Total_Desks_Sold__c',
                'Unomy_Alexa_Global_Rank__c',
                'Unomy_Alexa_US_Rank__c',
                'Won_Opportunities__c',
            ],
            'none':[]
        },
        'key_col': ['account_id'],
    },

    'location':{
        'not_feat_col':[
            'Unnamed: 0',
            'duns_number',
            'atlas_location_uuid',
            'longitude_loc',
            'latitude_loc',
            'city',
            'label',
        ],
        'dummy_col_name': ['City__c', 'Cluster__c', 'Contract_Status__c','Gate__c','HQ_by_WeWork__c', 'Market__c',
                          'NOT_Tourable_Sellable__c','Occupancy_Rating__c', 'Open_Studio__c', 'Portfolio_Standard_Name__c',
                         'Maximum_Tour_Days__c', 'Simultaneous_Tours__c', 'WeWork_Labs__c',  'Maximum_Tour_Days__c','Territory_Name__c',
                          'Time_Zone__c', 'Tour_Spacing__c'],
        'cont_col_name': {
            'norm': [
                'score_predicted_eo', 'score_employer', 'num_emp_weworkcore', 'num_poi_weworkcore',
               'pct_wwcore_employee', 'pct_wwcore_business', 'num_retail_stores', 'num_doctor_offices',
               'num_eating_places', 'num_drinking_places', 'num_hotels', 'num_fitness_gyms',
               'population_density', 'pct_female_population', 'median_age', 'income_per_capita',
               'pct_masters_degree', 'walk_score', 'bike_score', 'Total_Desks__c', 'Unavailable_Desks__c',
                'Unavailable_Offices__c', 'Total_Offices__c', 'Available_Desks__c','Available_Offices__c'
                   ]
            ,
            'minmax':[
                   'Desks_Occupied_Right_Now__c', 
                   'Desks_Ready_for_Occupany__c',
                   'Offices_Occupied_Right_Now__c',
                   'Offices_Ready_for_Occupany__c',  
                  ],
            'none':[
                'Occupancy_Rate__c'
            ]
        },
        'key_col': ['atlas_location_uuid'],
    }
}

