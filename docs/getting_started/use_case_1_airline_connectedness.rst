Use Case 1: Airline Regional Connections
========================================

Download this as a :download:`notebook </_downloads/notebooks/airtravel.tar.gz>`.

This is a tutorial on how to find the most well-connected regions of the
U.S. via air travel.

The U.S. Bureau of Transportation Statistics provides data on monthly
air travel from all certificated U.S. air carriers. The data can be
found
`here <https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=258>`__.

We utilized this data to determine which areas in the U.S. are most
well-connected.

We’ll first investigate which region is travelled through the most and
then which regions share the most travellers.

We chose 2018 data to avoid any impacts COVID-19 might’ve had on travel.

Data Preprocessing
==================

Let’s first look at the data.

First, we’ll need to import some libraries.

.. code:: python

    >>> import metagraph as mg
    >>> import pandas as pd

Let’s see what the data looks like.

.. code:: python

    >>> RAW_DATA_CSV = './raw_data.csv' # https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=258
    >>> raw_data_df = pd.read_csv(RAW_DATA_CSV)
    >>> raw_data_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>PASSENGERS</th>
          <th>FREIGHT</th>
          <th>MAIL</th>
          <th>DISTANCE</th>
          <th>UNIQUE_CARRIER</th>
          <th>AIRLINE_ID</th>
          <th>UNIQUE_CARRIER_NAME</th>
          <th>UNIQUE_CARRIER_ENTITY</th>
          <th>REGION</th>
          <th>CARRIER</th>
          <th>...</th>
          <th>DEST_STATE_ABR</th>
          <th>DEST_STATE_FIPS</th>
          <th>DEST_STATE_NM</th>
          <th>DEST_WAC</th>
          <th>YEAR</th>
          <th>QUARTER</th>
          <th>MONTH</th>
          <th>DISTANCE_GROUP</th>
          <th>CLASS</th>
          <th>Unnamed: 36</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0.0</td>
          <td>410.0</td>
          <td>0.0</td>
          <td>616.0</td>
          <td>WN</td>
          <td>19393.0</td>
          <td>Southwest Airlines Co.</td>
          <td>06725</td>
          <td>D</td>
          <td>WN</td>
          <td>...</td>
          <td>TN</td>
          <td>47</td>
          <td>Tennessee</td>
          <td>54</td>
          <td>2018</td>
          <td>2</td>
          <td>6</td>
          <td>2</td>
          <td>F</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>1</th>
          <td>0.0</td>
          <td>184.0</td>
          <td>0.0</td>
          <td>2592.0</td>
          <td>WN</td>
          <td>19393.0</td>
          <td>Southwest Airlines Co.</td>
          <td>06725</td>
          <td>D</td>
          <td>WN</td>
          <td>...</td>
          <td>CA</td>
          <td>6</td>
          <td>California</td>
          <td>91</td>
          <td>2018</td>
          <td>2</td>
          <td>6</td>
          <td>6</td>
          <td>F</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>2</th>
          <td>0.0</td>
          <td>87.0</td>
          <td>0.0</td>
          <td>2445.0</td>
          <td>WN</td>
          <td>19393.0</td>
          <td>Southwest Airlines Co.</td>
          <td>06725</td>
          <td>D</td>
          <td>WN</td>
          <td>...</td>
          <td>NY</td>
          <td>36</td>
          <td>New York</td>
          <td>22</td>
          <td>2018</td>
          <td>2</td>
          <td>6</td>
          <td>5</td>
          <td>F</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.0</td>
          <td>10.0</td>
          <td>0.0</td>
          <td>432.0</td>
          <td>WN</td>
          <td>19393.0</td>
          <td>Southwest Airlines Co.</td>
          <td>06725</td>
          <td>D</td>
          <td>WN</td>
          <td>...</td>
          <td>AR</td>
          <td>5</td>
          <td>Arkansas</td>
          <td>71</td>
          <td>2018</td>
          <td>2</td>
          <td>6</td>
          <td>1</td>
          <td>F</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>4</th>
          <td>0.0</td>
          <td>100.0</td>
          <td>0.0</td>
          <td>129.0</td>
          <td>WN</td>
          <td>19393.0</td>
          <td>Southwest Airlines Co.</td>
          <td>06725</td>
          <td>D</td>
          <td>WN</td>
          <td>...</td>
          <td>OR</td>
          <td>41</td>
          <td>Oregon</td>
          <td>92</td>
          <td>2018</td>
          <td>2</td>
          <td>6</td>
          <td>1</td>
          <td>F</td>
          <td>NaN</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>



A city market is a region that an airport supports. For example, New
York City has many airports (and it’s sometimes cheaper to fly into and
out of different airports), but all of their airports serve the same
region / city market.

Since we’re mostly concerned with where passengers will end up going
(and not which airport they choose), we will view city markets as the
regions of interest.

We will define a region as being well-connected if many people travel in
and out of it.

Let’s filter out all the irrelevant information not required for finding
the well-connected regions.

Let’s also filter out any flight paths with zero passengers (these
flights are usually flights transporting packages).

.. code:: python

    >>> RELEVANT_COLUMNS = [
    ...     'PASSENGERS',
    ...     'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', 'ORIGIN_STATE_ABR', 'ORIGIN_STATE_NM',
    ...     'DEST_AIRPORT_ID',   'DEST_AIRPORT_SEQ_ID',   'DEST_CITY_MARKET_ID',   'DEST',   'DEST_CITY_NAME',   'DEST_STATE_ABR',   'DEST_STATE_NM',
    ... ]
    ... 
    >>> relevant_df = raw_data_df[RELEVANT_COLUMNS]
    >>> relevant_df = relevant_df[relevant_df.PASSENGERS != 0.0]
    >>> relevant_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>PASSENGERS</th>
          <th>ORIGIN_AIRPORT_ID</th>
          <th>ORIGIN_AIRPORT_SEQ_ID</th>
          <th>ORIGIN_CITY_MARKET_ID</th>
          <th>ORIGIN</th>
          <th>ORIGIN_CITY_NAME</th>
          <th>ORIGIN_STATE_ABR</th>
          <th>ORIGIN_STATE_NM</th>
          <th>DEST_AIRPORT_ID</th>
          <th>DEST_AIRPORT_SEQ_ID</th>
          <th>DEST_CITY_MARKET_ID</th>
          <th>DEST</th>
          <th>DEST_CITY_NAME</th>
          <th>DEST_STATE_ABR</th>
          <th>DEST_STATE_NM</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>44447</th>
          <td>1.0</td>
          <td>12523</td>
          <td>1252306</td>
          <td>32523</td>
          <td>JNU</td>
          <td>Juneau, AK</td>
          <td>AK</td>
          <td>Alaska</td>
          <td>11545</td>
          <td>1154501</td>
          <td>31545</td>
          <td>ELV</td>
          <td>Elfin Cove, AK</td>
          <td>AK</td>
          <td>Alaska</td>
        </tr>
        <tr>
          <th>44448</th>
          <td>1.0</td>
          <td>12523</td>
          <td>1252306</td>
          <td>32523</td>
          <td>JNU</td>
          <td>Juneau, AK</td>
          <td>AK</td>
          <td>Alaska</td>
          <td>11619</td>
          <td>1161902</td>
          <td>31619</td>
          <td>EXI</td>
          <td>Excursion Inlet, AK</td>
          <td>AK</td>
          <td>Alaska</td>
        </tr>
        <tr>
          <th>44449</th>
          <td>1.0</td>
          <td>12610</td>
          <td>1261001</td>
          <td>32610</td>
          <td>KAE</td>
          <td>Kake, AK</td>
          <td>AK</td>
          <td>Alaska</td>
          <td>10204</td>
          <td>1020401</td>
          <td>30204</td>
          <td>AGN</td>
          <td>Angoon, AK</td>
          <td>AK</td>
          <td>Alaska</td>
        </tr>
        <tr>
          <th>44450</th>
          <td>1.0</td>
          <td>11298</td>
          <td>1129806</td>
          <td>30194</td>
          <td>DFW</td>
          <td>Dallas/Fort Worth, TX</td>
          <td>TX</td>
          <td>Texas</td>
          <td>11292</td>
          <td>1129202</td>
          <td>30325</td>
          <td>DEN</td>
          <td>Denver, CO</td>
          <td>CO</td>
          <td>Colorado</td>
        </tr>
        <tr>
          <th>44451</th>
          <td>1.0</td>
          <td>15991</td>
          <td>1599102</td>
          <td>35991</td>
          <td>YAK</td>
          <td>Yakutat, AK</td>
          <td>AK</td>
          <td>Alaska</td>
          <td>14828</td>
          <td>1482805</td>
          <td>34828</td>
          <td>SIT</td>
          <td>Sitka, AK</td>
          <td>AK</td>
          <td>Alaska</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>


We’ll want to have our data in an edge list format.

We’ll want the city markets to be the nodes.

Since we’re using betweenness centrality to determine connectedness,
we’ll want paths with less total weight to be the ones denoting paths
with more passengers. More elegant metrics might be considered in
practice, but we’ll use ``1/number_of_passengers`` for the weights for
our example.

We’ll create an edge list with such weights using pandas.

.. code:: python

    >>> passenger_flow_df = relevant_df[['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID', 'PASSENGERS']]
    >>> passenger_flow_df = passenger_flow_df.groupby(['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID']) \
    ...                         .PASSENGERS.sum() \
    ...                         .reset_index()
    ... 
    >>> passenger_flow_df['INVERSE_PASSENGER_COUNT'] = passenger_flow_df.PASSENGERS.map(lambda passenger_count: 1/passenger_count)
    >>> assert len(passenger_flow_df[passenger_flow_df.INVERSE_PASSENGER_COUNT != passenger_flow_df.INVERSE_PASSENGER_COUNT]) == 0, "Edge list has NaN weights."
    >>> passenger_flow_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ORIGIN_CITY_MARKET_ID</th>
          <th>DEST_CITY_MARKET_ID</th>
          <th>PASSENGERS</th>
          <th>INVERSE_PASSENGER_COUNT</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>30005</td>
          <td>30349</td>
          <td>4.0</td>
          <td>0.250000</td>
        </tr>
        <tr>
          <th>1</th>
          <td>30005</td>
          <td>31214</td>
          <td>10.0</td>
          <td>0.100000</td>
        </tr>
        <tr>
          <th>2</th>
          <td>30005</td>
          <td>31517</td>
          <td>193.0</td>
          <td>0.005181</td>
        </tr>
        <tr>
          <th>3</th>
          <td>30005</td>
          <td>35731</td>
          <td>7.0</td>
          <td>0.142857</td>
        </tr>
        <tr>
          <th>4</th>
          <td>30006</td>
          <td>30056</td>
          <td>5.0</td>
          <td>0.200000</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>



Since the data has city market IDs and don’t have names because an
airport can serve regions containing multiple cities, it’d be useful to
get a mapping from city market IDs to city names and airports.

.. code:: python

    >>> origin_city_market_id_info_df = relevant_df[['ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME']] \
    ...                                     .rename(columns={'ORIGIN_CITY_MARKET_ID': 'CITY_MARKET_ID',
    ...                                                      'ORIGIN': 'AIRPORT',
    ...                                                      'ORIGIN_CITY_NAME': 'CITY_NAME'})
    ... 
    >>> dest_city_market_id_info_df = relevant_df[['DEST_CITY_MARKET_ID', 'DEST', 'DEST_CITY_NAME']] \
    ...                                     .rename(columns={'DEST_CITY_MARKET_ID': 'CITY_MARKET_ID',
    ...                                                      'DEST': 'AIRPORT',
    ...                                                      'DEST_CITY_NAME': 'CITY_NAME'})
    ... 
    >>> city_market_id_info_df = pd.concat([origin_city_market_id_info_df, dest_city_market_id_info_df])
    >>> city_market_id_info_df = city_market_id_info_df.groupby('CITY_MARKET_ID').agg({'AIRPORT': set, 'CITY_NAME': set})
    >>> city_market_id_info_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>AIRPORT</th>
          <th>CITY_NAME</th>
        </tr>
        <tr>
          <th>CITY_MARKET_ID</th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>30005</th>
          <td>{05A}</td>
          <td>{Little Squaw, AK}</td>
        </tr>
        <tr>
          <th>30006</th>
          <td>{06A}</td>
          <td>{Kizhuyak, AK}</td>
        </tr>
        <tr>
          <th>30007</th>
          <td>{KLW}</td>
          <td>{Klawock, AK}</td>
        </tr>
        <tr>
          <th>30009</th>
          <td>{09A, HOM}</td>
          <td>{Homer, AK}</td>
        </tr>
        <tr>
          <th>30010</th>
          <td>{1B1}</td>
          <td>{Hudson, NY}</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>



Which region is travelled through the most?
===========================================

We’re going to determine which region is travelled through the most
using Betweenness Centrality as it measures exactly that. There are a
variety of algorithms to choose from, but we’ll stick to using solely
Betweenness Centrality for this tutorial.

We’ll first create a metagraph graph for the data.

.. code:: python

    >>> r = mg.resolver
    >>> passenger_flow_graph_wrapped = r.wrappers.EdgeMap.PandasEdgeMap(passenger_flow_df, 
    ...                                                                 'ORIGIN_CITY_MARKET_ID', 
    ...                                                                 'DEST_CITY_MARKET_ID', 
    ...                                                                 'INVERSE_PASSENGER_COUNT',
    ...                                                                 is_directed=True)

Note that we use the inverse passenger count as the weights to ensure
that the shortest paths are the paths that have the most passengers.

Let’s calculate the Betweenness Centrality.

.. code:: python

    >>> betweenness_centrality = r.algos.vertex_ranking.betweenness_centrality(passenger_flow_graph_wrapped, 100, False, False)


Let’s look at the results and find the highest scores (which would give
us the city market IDs that are most travelled through).

.. code:: python

    >>> number_of_best_scores = 15
    >>> best_betweenness_centrality_scores = sorted(betweenness_centrality.value.items(), key=lambda x: x[1], reverse=True)[:number_of_best_scores]
    >>> best_betweenness_centrality_scores




.. parsed-literal::

    [(32575, 46411.0),
     (30559, 44345.0),
     (30299, 38214.0),
     (31703, 31860.0),
     (30977, 25167.0),
     (30397, 14147.0),
     (30194, 13684.0),
     (31517, 9340.0),
     (30070, 7826.0),
     (30325, 7593.0),
     (30113, 7585.0),
     (32467, 7459.0),
     (32721, 5966.0),
     (30154, 5885.0),
     (30056, 5878.0)]



Now that we have the city market IDs with the best scores, let’s find
out which regions those city market IDs correspond to using the mapping
from city market IDs to city names and airports we made earlier.

.. code:: python

    >>> best_betweenness_centrality_scores_df = pd.DataFrame(best_betweenness_centrality_scores).rename(columns={0:'CITY_MARKET_ID', 1:'BETWEENNESS_CENTRALITY_SCORE'}).set_index('CITY_MARKET_ID')
    >>> best_betweenness_centrality_scores_df.join(city_market_id_info_df).sort_values('BETWEENNESS_CENTRALITY_SCORE', ascending=False)




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>BETWEENNESS_CENTRALITY_SCORE</th>
          <th>AIRPORT</th>
          <th>CITY_NAME</th>
        </tr>
        <tr>
          <th>CITY_MARKET_ID</th>
          <th></th>
          <th></th>
          <th></th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>32575</th>
          <td>46411.0</td>
          <td>{BUR, HHR, SMO, ONT, VNY, LAX, SNA, LGB}</td>
          <td>{Burbank, CA, Long Beach, CA, Hawthorne, CA, O...</td>
        </tr>
        <tr>
          <th>30559</th>
          <td>44345.0</td>
          <td>{LKE, KEH, BFI, SEA}</td>
          <td>{Kenmore, WA, Seattle, WA}</td>
        </tr>
        <tr>
          <th>30299</th>
          <td>38214.0</td>
          <td>{MRI, ANC, DQL}</td>
          <td>{Anchorage, AK}</td>
        </tr>
        <tr>
          <th>31703</th>
          <td>31860.0</td>
          <td>{TSS, JFK, JRA, EWR, SWF, JRB, HPN, LGA, ISP}</td>
          <td>{Newark, NJ, Newburgh/Poughkeepsie, NY, Islip,...</td>
        </tr>
        <tr>
          <th>30977</th>
          <td>25167.0</td>
          <td>{PWK, ORD, GYY, LOT, MDW, DPA}</td>
          <td>{Chicago, IL, Chicago/Romeoville, IL, Gary, IN}</td>
        </tr>
        <tr>
          <th>30397</th>
          <td>14147.0</td>
          <td>{ATL, PDK, QMA, FTY}</td>
          <td>{Atlanta, GA, Kennesaw, GA}</td>
        </tr>
        <tr>
          <th>30194</th>
          <td>13684.0</td>
          <td>{DFW, ADS, DAL, AFW, FTW, RBD, FWH}</td>
          <td>{Fort Worth, TX, Dallas, TX, Dallas/Fort Worth...</td>
        </tr>
        <tr>
          <th>31517</th>
          <td>9340.0</td>
          <td>{FAI, MTX, FBK, EIL, A01}</td>
          <td>{Fairbanks/Ft. Wainwright, AK, Fairbanks, AK}</td>
        </tr>
        <tr>
          <th>30070</th>
          <td>7826.0</td>
          <td>{ADQ, KDK}</td>
          <td>{Kodiak, AK}</td>
        </tr>
        <tr>
          <th>30325</th>
          <td>7593.0</td>
          <td>{APA, DEN}</td>
          <td>{Denver, CO}</td>
        </tr>
        <tr>
          <th>30113</th>
          <td>7585.0</td>
          <td>{BET}</td>
          <td>{Bethel, AK}</td>
        </tr>
        <tr>
          <th>32467</th>
          <td>7459.0</td>
          <td>{OPF, MIA, FLL, TMB, MPB, FXE}</td>
          <td>{Miami, FL, Fort Lauderdale, FL}</td>
        </tr>
        <tr>
          <th>32721</th>
          <td>5966.0</td>
          <td>{KLN}</td>
          <td>{Larsen Bay, AK}</td>
        </tr>
        <tr>
          <th>30154</th>
          <td>5885.0</td>
          <td>{ACK}</td>
          <td>{Nantucket, MA}</td>
        </tr>
        <tr>
          <th>30056</th>
          <td>5878.0</td>
          <td>{A43}</td>
          <td>{Kodiak Island, AK}</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>



This is what we’d expect. Highly populated areas like Los Angeles are
the most traveled through areas.

However, it’s surprising that Anchorage is more travelled through than a
hub like Dallas!

There’s a good explanation for Anchorage being a very travelled through
region: Since Alaska is so sparsely populated, a well-connected road
infrastructure was never built. Thus, to travel between cities in
Alaska, air travel is the only option. More information can be found
`here <https://en.wikipedia.org/wiki/List_of_airports_in_Alaska>`__.
