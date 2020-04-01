import pandas as pd
import math

class Sampler:
    def __init__(self, sources):
        building_df = sources['building']
        op_df = sources['opportunity']
        geo_df = sources['geography']
        us_building_df = building_df[(building_df['country'] == 'USA') &
                                    (~building_df['atlas_location_uuid'].isna()) & 
                                    (building_df['atlas_location_uuid'] != 'TestBuilding')]
        us_building_df = us_building_df.drop(columns='country')
        us_op_building_df = op_df.merge(us_building_df, on='atlas_location_uuid')
        us_op_df = us_op_building_df[['account_id', 'atlas_location_uuid', 'city']]
        us_op_city_df = us_op_df.drop(columns='atlas_location_uuid')
        us_op_city_df = us_op_city_df.drop_duplicates(['account_id', 'city'], keep='last')
        #us_op_atlas_df = us_op_city_df.merge(us_building_df, on='city')
        #us_building_df = us_building_df.rename(columns={'atlas_location_uuid': 'atlas_location_uuid_2nd'})
#         self.us_op_atlas_df = us_op_atlas_df.drop(columns='city')
        self.us_op_city_df = us_op_city_df
        self.us_building_geo_df = us_building_df.merge(geo_df, on='geo_id')
        self.us_op_df = us_op_df
        # cities = ['San Jose', 'Seattle']
        # self.us_op_city_df = self.us_op_city_df[self.us_op_city_df['city'].isin(cities) ]
        # self.us_building_geo_df = self.us_building_geo_df[self.us_building_geo_df['city'].isin(cities)]
        # self.us_op_df = self.us_op_df[self.us_op_df['city'].isin(cities)]
    
    def pairwise_distances(self, op_df):
        locs = []
        neu_pairs = []
        neg_pairs = []
        building_geo_df = self.us_building_geo_df
        for index, row in building_geo_df.iterrows():
            locs.append((row['atlas_location_uuid'], row['long'], row['lat'], row['city']))
        locs_len = len(locs)
        for i in range(locs_len):
            for j in range(i+1,locs_len):
                loc1, loc2 = locs[i], locs[j]
                city1, city2 = loc1[3], loc2[3]
                if city1 != city2: continue
                distance = self.geo_distance(loc1[1], loc2[1], loc1[2], loc2[2])
                if distance < 5.0:
                    neu_pairs.append((loc1[0], loc2[0], distance))
                    neu_pairs.append((loc2[0], loc1[0], distance))
                elif distance < 50.0:
                    neg_pairs.append((loc1[0], loc2[0], distance))
                    neg_pairs.append((loc2[0], loc1[0], distance))

        geo_neg_df = pd.DataFrame(neg_pairs, columns={'atlas_location_uuid', 'atlas_location_uuid_near', 'distance'})
        geo_neu_df = pd.DataFrame(neu_pairs, columns={'atlas_location_uuid', 'atlas_location_uuid_near', 'distance'})
        return geo_neu_df, geo_neg_df
    
    def account_level_visit(self, op_df):
        acc = op_df.name
        geo_neu_df =  self.geo_neu_df
        us_op_df = self.us_op_df
        vis_locs = us_op_df[us_op_df['account_id'] == acc]['atlas_location_uuid'].unique().tolist()
        geo_neu_df = geo_neu_df[geo_neu_df['atlas_location_uuid'].isin(vis_locs)]
        vis_near_locs = geo_neu_df['atlas_location_uuid_near'].unique().tolist()
        interest_locs = set()
        for loc in vis_locs + vis_near_locs:
            interest_locs.add(loc)
        neg_df = op_df[~op_df['atlas_location_uuid'].isin(interest_locs)]
        if len(neg_df) > 0:
            ratio = ((len(vis_locs)*1.0 / len(neg_df)) / (1.0 / 3))
            if ratio < 1.0:
                neg_df = neg_df.sample(frac=ratio).reset_index(drop=True)
        return neg_df
    
    def city_level_visit(self, op_df):
        city = op_df.name
        city_building_geo_df = self.us_building_geo_df[self.us_building_geo_df['city'] == city]
        op_city_df = self.us_op_city_df[self.us_op_city_df['city']==city]
        all_op_df = op_city_df.merge(city_building_geo_df, on='city', suffixes=('op', 'building'))
        neg_df = all_op_df.groupby('account_id').apply(self.account_level_visit) 
        return neg_df

    def negative_sampling(self):
        #negative_location_ids = set.union(*[self.item_pool['city_loc'][self.item_pool['loc_city'][x]] for x in location_ids])
        op_df = self.us_op_df
        self.geo_neu_df, self.geo_neg_df = self.pairwise_distances(op_df)
        neg_op_df = op_df.groupby('city').apply(self.city_level_visit)
        return neg_op_df

    def positive_sampling(self):
        return self.us_op_df

    def geo_distance(self, lon1,lon2,lat1,lat2):
        try:
            radius = 6371 # km
            dlat = math.radians(lat2-lat1)
            dlon = math.radians(lon2-lon1)
            a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
                * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            d = radius * c
        except:
            d = 1000
        return d