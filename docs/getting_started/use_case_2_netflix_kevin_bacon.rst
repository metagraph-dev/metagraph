Use Case 2: Kevin Bacon of Neflix 2019
======================================

Download this as a :download:`notebook </_static/notebooks/kevin_bacon.tar.gz>`.

This is a tutorial on how to find the most well-connected Netflix cast
member of 2019.

Bacon’s Law is a concept claiming that most people in the Hollywood
film industry can be linked through their film roles to Kevin Bacon
within six steps.

We’ll go over how to find out who are the centers of the the Netflix
film world, similar to how Bacon is the center of the Hollywood
film industry.

We'll use a Kaggle dataset containing all the TV shows and movies on
Netflix as of 2019. The dataset can be found
`here <https://www.kaggle.com/shivamb/netflix-shows>`__.

Preprocess Data
===============

The raw data is in a tabular format with columns for movies, cast
members, directors, release dates, countries of release, etc.

We’ll want to put it in a graph-friendly format. In particular, we’ll
want to convert it to an edge list format.

First, we’ll import some necessary libraries.

.. code:: python

    >>> import os
    >>> import pandas as pd
    >>> import networkx as nx
    >>> import metagraph as mg
    >>> import scipy.sparse as ss
    >>> from collections import Counter
    >>> from typing import Union

Let’s take a look at the raw data provided.

.. code:: python

    >>> RAW_DATA_CSV = './netflix_titles.csv' # https://www.kaggle.com/shivamb/netflix-shows
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
	 
        .dataframe table {
	    width: 12000px;
	}
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>show_id</th>
          <th>type</th>
          <th>title</th>
          <th>director</th>
          <th>cast</th>
          <th>country</th>
          <th>date_added</th>
          <th>release_year</th>
          <th>rating</th>
          <th>duration</th>
          <th>listed_in</th>
          <th>description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>81145628</td>
          <td>Movie</td>
          <td>Norm of the North: King Sized Adventure</td>
          <td>Richard Finn, Tim Maltby</td>
          <td>Alan Marriott, Andrew Toth, Brian Dobson, Cole...</td>
          <td>United States, India, South Korea, China</td>
          <td>September 9, 2019</td>
          <td>2019</td>
          <td>TV-PG</td>
          <td>90 min</td>
          <td>Children &amp; Family Movies, Comedies</td>
          <td>Before planning an awesome wedding for his gra...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>80117401</td>
          <td>Movie</td>
          <td>Jandino: Whatever it Takes</td>
          <td>NaN</td>
          <td>Jandino Asporaat</td>
          <td>United Kingdom</td>
          <td>September 9, 2016</td>
          <td>2016</td>
          <td>TV-MA</td>
          <td>94 min</td>
          <td>Stand-Up Comedy</td>
          <td>Jandino Asporaat riffs on the challenges of ra...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>70234439</td>
          <td>TV Show</td>
          <td>Transformers Prime</td>
          <td>NaN</td>
          <td>Peter Cullen, Sumalee Montano, Frank Welker, J...</td>
          <td>United States</td>
          <td>September 8, 2018</td>
          <td>2013</td>
          <td>TV-Y7-FV</td>
          <td>1 Season</td>
          <td>Kids' TV</td>
          <td>With the help of three human allies, the Autob...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>80058654</td>
          <td>TV Show</td>
          <td>Transformers: Robots in Disguise</td>
          <td>NaN</td>
          <td>Will Friedle, Darren Criss, Constance Zimmer, ...</td>
          <td>United States</td>
          <td>September 8, 2018</td>
          <td>2016</td>
          <td>TV-Y7</td>
          <td>1 Season</td>
          <td>Kids' TV</td>
          <td>When a prison ship crash unleashes hundreds of...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>80125979</td>
          <td>Movie</td>
          <td>#realityhigh</td>
          <td>Fernando Lebrija</td>
          <td>Nesta Cooper, Kate Walsh, John Michael Higgins...</td>
          <td>United States</td>
          <td>September 8, 2017</td>
          <td>2017</td>
          <td>TV-14</td>
          <td>99 min</td>
          <td>Comedies</td>
          <td>When nerdy high schooler Dani finally attracts...</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>

We’ll only consider movies since multiple cast members can work on the
same TV show but may not ever see each other on set.

We’ll also only consider U.S. movies since cast members from different
countries often do not work together.

We’ll necessarily need to remove any rows with missing data as well.

.. code:: python

    >>> movies_df = raw_data_df[raw_data_df['type']=='Movie'].drop(columns=['type']).dropna()
    >>> movies_df = movies_df[movies_df.country.str.contains('United States')]
    >>> movies_df.head()

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
          <th>show_id</th>
          <th>title</th>
          <th>director</th>
          <th>cast</th>
          <th>country</th>
          <th>date_added</th>
          <th>release_year</th>
          <th>rating</th>
          <th>duration</th>
          <th>listed_in</th>
          <th>description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>81145628</td>
          <td>Norm of the North: King Sized Adventure</td>
          <td>Richard Finn, Tim Maltby</td>
          <td>Alan Marriott, Andrew Toth, Brian Dobson, Cole...</td>
          <td>United States, India, South Korea, China</td>
          <td>September 9, 2019</td>
          <td>2019</td>
          <td>TV-PG</td>
          <td>90 min</td>
          <td>Children &amp; Family Movies, Comedies</td>
          <td>Before planning an awesome wedding for his gra...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>80125979</td>
          <td>#realityhigh</td>
          <td>Fernando Lebrija</td>
          <td>Nesta Cooper, Kate Walsh, John Michael Higgins...</td>
          <td>United States</td>
          <td>September 8, 2017</td>
          <td>2017</td>
          <td>TV-14</td>
          <td>99 min</td>
          <td>Comedies</td>
          <td>When nerdy high schooler Dani finally attracts...</td>
        </tr>
        <tr>
          <th>6</th>
          <td>70304989</td>
          <td>Automata</td>
          <td>Gabe Ibáñez</td>
          <td>Antonio Banderas, Dylan McDermott, Melanie Gri...</td>
          <td>Bulgaria, United States, Spain, Canada</td>
          <td>September 8, 2017</td>
          <td>2014</td>
          <td>R</td>
          <td>110 min</td>
          <td>International Movies, Sci-Fi &amp; Fantasy, Thrillers</td>
          <td>In a dystopian future, an insurance adjuster f...</td>
        </tr>
        <tr>
          <th>9</th>
          <td>70304990</td>
          <td>Good People</td>
          <td>Henrik Ruben Genz</td>
          <td>James Franco, Kate Hudson, Tom Wilkinson, Omar...</td>
          <td>United States, United Kingdom, Denmark, Sweden</td>
          <td>September 8, 2017</td>
          <td>2014</td>
          <td>R</td>
          <td>90 min</td>
          <td>Action &amp; Adventure, Thrillers</td>
          <td>A struggling couple can't believe their luck w...</td>
        </tr>
        <tr>
          <th>11</th>
          <td>70299204</td>
          <td>Kidnapping Mr. Heineken</td>
          <td>Daniel Alfredson</td>
          <td>Jim Sturgess, Sam Worthington, Ryan Kwanten, A...</td>
          <td>Netherlands, Belgium, United Kingdom, United S...</td>
          <td>September 8, 2017</td>
          <td>2015</td>
          <td>R</td>
          <td>95 min</td>
          <td>Action &amp; Adventure, Dramas, International Movies</td>
          <td>When beer magnate Alfred "Freddy" Heineken is ...</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>


All the cast members for a movie are in the same cell.

To have the data in an edge list format, we’ll need to use Pandas to
reformat the data to have rows where each cast member cell contains
exactly one cast member. This will mean that a movie will have multiple
rows (one for each cast member).

.. code:: python

    >>> def expand_dataframe_list_values_for_column(df: pd.DataFrame, column_name: Union[str, int]) -> pd.DataFrame:
            return df.apply(lambda x: pd.Series(x[column_name].split(', ')), axis=1) \
                          .stack() \
                          .reset_index(level=1, drop=True) \
                          .to_frame(column_name) \
                          .join(df.drop(columns=[column_name]))
        
    >>> movies_df = expand_dataframe_list_values_for_column(movies_df, 'cast')
    >>> movies_df.head()

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
          <th>cast</th>
          <th>show_id</th>
          <th>title</th>
          <th>director</th>
          <th>country</th>
          <th>date_added</th>
          <th>release_year</th>
          <th>rating</th>
          <th>duration</th>
          <th>listed_in</th>
          <th>description</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Alan Marriott</td>
          <td>81145628</td>
          <td>Norm of the North: King Sized Adventure</td>
          <td>Richard Finn, Tim Maltby</td>
          <td>United States, India, South Korea, China</td>
          <td>September 9, 2019</td>
          <td>2019</td>
          <td>TV-PG</td>
          <td>90 min</td>
          <td>Children &amp; Family Movies, Comedies</td>
          <td>Before planning an awesome wedding for his gra...</td>
        </tr>
        <tr>
          <th>0</th>
          <td>Andrew Toth</td>
          <td>81145628</td>
          <td>Norm of the North: King Sized Adventure</td>
          <td>Richard Finn, Tim Maltby</td>
          <td>United States, India, South Korea, China</td>
          <td>September 9, 2019</td>
          <td>2019</td>
          <td>TV-PG</td>
          <td>90 min</td>
          <td>Children &amp; Family Movies, Comedies</td>
          <td>Before planning an awesome wedding for his gra...</td>
        </tr>
        <tr>
          <th>0</th>
          <td>Brian Dobson</td>
          <td>81145628</td>
          <td>Norm of the North: King Sized Adventure</td>
          <td>Richard Finn, Tim Maltby</td>
          <td>United States, India, South Korea, China</td>
          <td>September 9, 2019</td>
          <td>2019</td>
          <td>TV-PG</td>
          <td>90 min</td>
          <td>Children &amp; Family Movies, Comedies</td>
          <td>Before planning an awesome wedding for his gra...</td>
        </tr>
        <tr>
          <th>0</th>
          <td>Cole Howard</td>
          <td>81145628</td>
          <td>Norm of the North: King Sized Adventure</td>
          <td>Richard Finn, Tim Maltby</td>
          <td>United States, India, South Korea, China</td>
          <td>September 9, 2019</td>
          <td>2019</td>
          <td>TV-PG</td>
          <td>90 min</td>
          <td>Children &amp; Family Movies, Comedies</td>
          <td>Before planning an awesome wedding for his gra...</td>
        </tr>
        <tr>
          <th>0</th>
          <td>Jennifer Cameron</td>
          <td>81145628</td>
          <td>Norm of the North: King Sized Adventure</td>
          <td>Richard Finn, Tim Maltby</td>
          <td>United States, India, South Korea, China</td>
          <td>September 9, 2019</td>
          <td>2019</td>
          <td>TV-PG</td>
          <td>90 min</td>
          <td>Children &amp; Family Movies, Comedies</td>
          <td>Before planning an awesome wedding for his gra...</td>
        </tr>
      </tbody>
    </table>
    </div>
    <br/>

.. code:: python

    >>> len(movies_df)
    
    13317



Now that we have the data in an edgelist format (where edges connect
cast members to movies) we want to put the data into a graph format.
We’ll use NetworkX.

.. code:: python

    movies_graph = nx.from_pandas_edgelist(movies_df, 'cast', 'title')

Note that the above graph is a bipartite graph of cast members and
movies. Since we want a graph where the edges connect actors who’ve
worked together on a movie, we’ll use NetworkX’s bipartite graph
projection functionality to generate this graph.

.. code:: python

    >>> actors = movies_df.cast.unique()
    >>> actor_graph = nx.projected_graph(movies_graph, actors)
    >>> len(actor_graph.nodes)
    
    8670
    
    >>> len(actor_graph.edges)
    
    63502

Note that the graph we generated has fewer edges than our data had rows.
This is because many pairs of actors have worked on multiple movies
together.

Find The Kevin Bacon(s)
=======================

We’re going to find the Kevin Bacons.

We’ll refer to the maximum number of hops a cast member needs to reach
all other cast members as the “Kevin Bacon distance”.

The Kevin Bacons are the cast members who have the smallest Kevin Bacon
distance.

To find the Kevin Bacons, we’ll first have to find all the connected
components (since we don’t exactly have a Kevin Bacon if our graph is
disconnected).

.. code:: python

    >>> r = mg.resolver
    >>> actor_graph_wrapped = r.wrappers.EdgeSet.NetworkXEdgeSet(actor_graph)
    >>> cc_node_label_mapping_wrapped = r.algos.clustering.connected_components(actor_graph_wrapped)
    >>> cc_node_label_mapping = cc_node_label_mapping_wrapped.value
    >>> label_counts = Counter()
    >>> for _, label in cc_node_label_mapping.items():
    ...     label_counts[label] += 1
    ... 
    >>> label_counts

.. parsed-literal::

    Counter({0: 7833,
             1: 10,
             2: 1,
             3: 1,
             4: 10,
             5: 1,
             6: 1,
             7: 1,
             8: 2,
             9: 10,
             10: 3,
             11: 1,
             12: 1,
             13: 10,
	     ... })



We have multiple connected components. We will find the Kevin Bacon of
the largest connected component since that one has more edges between
actors.

.. code:: python

    >>> largest_cc_label, _ = max(label_counts.items(), key = lambda pair: pair[1])
    >>> largest_cc_node_set = {node for node, label in cc_node_label_mapping.items() if label == largest_cc_label}
    >>> largest_cc_node_set_wrapped = r.wrappers.NodeSet.PythonNodeSet(largest_cc_node_set)
    >>> largest_cc_subgraph_wrapped = r.algos.subgraph.extract_edgeset(actor_graph_wrapped, largest_cc_node_set_wrapped)

We now need to find each actor’s Kevin Bacon distance.

One of the benefits of using metagraph is that when we are not forced to use metagraph end-to-end since metagraph supports algorithms on graph with various internal representations.

We can take out graph easily out of metagraph and use whatever graph library we desire because of metagraph's translation capabilities.

We'll demonstrate how to take our NetworkX graph, convert it into a SciPy adjacency matrix, and run SciPy's implementation of Dijkstra on it.

.. code:: python

    >>> largest_cc_subgraph_wrapped = r.translate(largest_cc_subgraph_wrapped, r.wrappers.EdgeSet.ScipyEdgeSet)
    >>> distance_matrix = ss.csgraph.dijkstra(largest_cc_subgraph_wrapped.value)

Once we have all the Kevin Bacon distances from every cast member, we can find the smallest Kevin Bacon distance.

.. code:: python

    >>> kevin_bacon_dists = distance_matrix.max(axis=0)
    >>> min_kevin_bacon_dist = kevin_bacon_dists.min()
    >>> min_kevin_bacon_dist

    6.0


From here, we can determine the Kevin Bacon(s)!

.. code:: python

    >>> kevin_bacon_indices = np.where(kevin_bacon_dists==min_kevin_bacon_dist)[0]
    >>> kevin_bacons = sorted([largest_cc_subgraph_wrapped.node_list[kevin_bacon_index] 
                               for kevin_bacon_index in kevin_bacon_indices])
    >>> len(kevin_bacons)

    295

    >>> print(sorted(kevin_bacons))

    [50 Cent, Aasif Mandvi, Adam Pally, Adam Scott, Alec Baldwin, Alexis Bledel,
     Alfred Molina, Alison Pill, Amanda Plummer, America Ferrera, Andrew Bachelor,
     Andy Richter, Andy Samberg, Angelique Cabral, Anna Faris, Anna Kendrick,
     Anthony Anderson, Anthony Hopkins, Anthony Mackie, Beau Bridges, Ben Kingsley,
     Benicio Del Toro, Bill Murray, Billy Connolly, Bob Odenkirk, Bobby Cannavale,
     Bradley Cooper, Brandon Routh, Brian Tyree Henry, Brie Larson, Brittany Murphy,
     Brooke D'Orsay, Brooklyn Decker, Bruce Willis, Busy Philipps, Cameron Diaz,
     Cate Blanchett, Cathy Cliften, Celia Weston, Charlie Murphy, Charlie Sheen,
     Chelcie Ross, Chloë Grace Moretz, Chris Parnell, Chris Pratt, Christina Hendricks,
     Christina Ricci, Christopher McDonald, Christopher Mintz-Plasse, Christopher Plummer,
     Christopher Walken, Clive Owen, Cole Hauser, Common, Constance Wu, Danai Gurira,
     Danny Trejo, David Koechner, Dax Shepard, Demi Moore, Dennis Quaid, Dermot Mulroney,
     Devon Aoki, Diane Keaton, Djimon Hounsou, Don Cheadle, Donal Logue, Donald Faison,
     Dwayne Johnson, Elijah Wood, Elizabeth Banks, Elizabeth Perkins, Ellen Barkin,
     Emily Watson, Emma Roberts, Emma Stone, Emmy Rossum, Eric Stoltz, Finesse Mitchell,
     Fionnula Flanagan, Forest Whitaker, Frank Grillo, Frank Langella, Fred Armisen,
     Gary Cole, Gary Oldman, Geraldine James, Gerard Butler, Gina Gershon,
     Giovanni Ribisi, Greg Kinnear, Gugu Mbatha-Raw, Halle Berry, Hank Azaria,
     Harry Connick Jr., Harvey Keitel, Heather Graham, Hilary Swank, Hugh Jackman,
     Hugh Laurie, Ian McShane, Iko Uwais, J.B. Smoove, J.K. Simmons, Jack McBrayer,
     Jack Nicholson, Jacki Weaver, Jaime King, James Caan, James Marsden, James Remar,
     Jane Curtin, Janeane Garofalo, Jared Leto, Jason Butler Harner, Jason Sudeikis,
     Jay Hernandez, Jay Mohr, Jeffrey Tambor, Jennifer Coolidge, Jennifer Garner,
     Jennifer Jason Leigh, Jesse Williams, Jessica Alba, Jessica Biel, Jessica Simpson,
     Jessica Szohr, Jim Carrey, Jim Parsons, Jim Sturgess, Joan Cusack, Joanna Going,
     Joe Torry, Joey King, John Beasley, John C. Reilly, John Cleese, John Cusack,
     John Hodgman, John Leguizamo, John Michael Higgins, John Travolta,
     Johnny Knoxville, Jon Voight, Jonah Hill, Josh Brolin, Josh Duhamel, Josh Gad,
     Josh Hartnett, Julianne Moore, Justin Long, Justin Timberlake, Kate Berlant,
     Kate Bosworth, Kate Winslet, Kathleen Chalfant, Katie Holmes, Keanu Reeves,
     Kellita Smith, Kelsey Grammer, Kevin Bacon, Kevin Costner, Kieran Culkin,
     Kim Dickens, Kirsten Dunst, Kristen Bell, Kristin Chenoweth, Lauren Graham,
     Laurence Fishburne, Leila Arcieri, Leslie Bibb, Liev Schreiber, Lili Taylor,
     Lin Shaye, Lindsay Burdge, Loretta Devine, Louis C.K., Louisa Krause,
     Lucien Laviscount, Lynn Collins, Macon Blair, Mae Whitman, Maria de Medeiros,
     Mark Blum, Mark Webber, Mary Alice, Mary Elizabeth Winstead, Matt Dillon,
     Matt Walsh, Matthew Goode, Maya Rudolph, Meagan Good, Melanie Lynskey,
     Melissa Leo, Michael Clarke Duncan, Michael Jeter, Michael Madsen, Michael Sheen,
     Mickey Rourke, Mike Epps, Mike Myers, Mike Vogel, Molly Shannon, Monica Bellucci,
     Naomi Watts, Natalie Martinez, Nick Stahl, Nicolas Cage, Nicole Ari Parker,
     Nicole Byer, Olivia Holt, Olivia Munn, Omid Djalili, Oprah Winfrey, Pat Healy,
     Patrick Stewart, Patrick Warburton, Patrick Wilson, Paul Rudd, Phil Crowley,
     Pierce Brosnan, Powers Boothe, Queen Latifah, Randall Park, Ray Liotta,
     Regina Hall, Renée Zellweger, Retta, Richard Gere, Richard Jenkins, Rick Yune,
     Ricky Gervais, Rob Lowe, Robert Forster, Robert Patrick, Romany Malco,
     Ron Livingston, Rosanna Arquette, Rosario Dawson, Rose Byrne, Rosemary Harris,
     Rotimi, Russell Brand, Russell Simmons, Rutger Hauer, Ryan Hansen, Ryan Kwanten,
     Ryan Phillippe, Ryan Reynolds, Sam Page, Sam Worthington, Samuel L. Jackson,
     Sarah Jessica Parker, Sarah Shahi, Sarah Silverman, Sean Hayes, Seann William Scott,
     Sebastian Stan, Seth Green, Seth MacFarlane, Seth Meyers, Sheryl Underwood,
     Shirley MacLaine, Stephen Merchant, Stephen Root, Steve Buscemi, Susie Essman,
     T.I., T.J. Miller, Ted Danson, Terence Stamp, Terrence Howard, Terry Crews,
     Tim Roth, Tina Fey, Tom Arnold, Tom Hollander, Tommy Lee Jones, Tony Darrow,
     Uma Thurman, Val Kilmer, Verne Troyer, Victor Garber, Ving Rhames, Viola Davis,
     Wesley Snipes, Will Ferrell, Will Forte, Will Sasso, Willem Dafoe, Yolonda Ross,
     Yul Vazquez, Zach Braff, Zoe Saldana, Zoë Kravitz]

It turns out that the Netflix world is much more connected than the whole Hollywood film industry since there are so many cast members who are well-connected.
