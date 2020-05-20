import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import uuid

df = pd.read_csv('salesforce_opp.csv', usecols=['account_id', 'opportunity_id', 'end_date', 'last_modified_date'])
df.end_date.fillna(df.last_modified_date, inplace=True)
df.drop(columns='last_modified_date', inplace=True)

def compute_buying_frequencies(df):
  df['end_date'] = pd.to_datetime(df.end_date)
  df.sort_values(['account_id', 'end_date'], inplace=True)
  df['buying_frequency'] = df.groupby('account_id').end_date.apply(lambda x: x.diff())
  df_account_first_opp = df.loc[df.buying_frequency.isna()].copy(deep=True) #saving first opportunities
  df.dropna(axis=0, inplace=True) #dropping first opportunities
  return df

df = compute_buying_frequencies(df.copy(deep=True))

def find_optimal_k(df):
  sse = pd.DataFrame(columns=['inertia','k'])
  for k in range(2, 20):
    kmeans = KMeans(n_cluster=k)
    kmeans.fit(df.buying_frequency.values)
    sse = sse.append({'inertia': kmeans.inertia_,'k': k}, ignore_index=True) # Inertia: Sum of distances of samples to their closest cluster center
  plt.figure()
  plt.bar(data=sse, x='k', y='inertia')
  plt.show()

find_optimal_k(df)

def predicting_buying_frequency_cluster(df, k):
  kmeans = KMeans(k=k).fit(df.buying_frequency.values)
  df['buying_frequency_cluster'] = kmeans.predict(df.buying_frequency.values)
  return df

k = 7 #to be changed based on your results
df = predicting_buying_frequency_cluster(df.copy(deep=True), k)

def renaming_cluster(df):
  df['mean_cluster'] = df.groupby('buying_frequency_cluster')['buying_frequency'].transform('mean')
  temp = pd.DataFrame(df.mean_cluster.unique(), columns=['mean_cluster']).sort_values('mean_cluster', ascending=False).reset_index(drop=True)
  temp['cluster_'+col] = temp.index
  df = pd.merge(df.drop(columns='buying_frequency_cluster'), temp, on='mean_cluster')
  return df

df = renaming_cluster(df.copy(deep=True))

def flagging_lapsing_period(df):
  df['last_opp_cluster_mean'] = df.groupby('account_id').mean_cluster.apply(lambda x: shift(1))
  flag_opportunities = df.loc[df.change_to_slower_buying_frequency].copy(deep=True)
  df = df.loc[~df.change_to_slower_buying_frequency]
  return df, flag_opportunities
  
df, flag_opportunities = flagging_lapsing_period(df.copy(deep=True))
  
def addding_missed_opportunities_during_lapsing_period(df, flag_opportunities):
  flag_opportunities['nb_missed_opp_during_lapsing_period'] = flag_opportunities.apply(axis=1, \
                                                              func=lambda x: int(x['buying_frequency']/x['last_opp_cluster_mean']))                                                                                                                      
  max_nb_missed_opp = flag_opportunties['number_of_missed_opportunities_during_lapsing_period'].max()
  df = pd.concat([df, flag_opportunies.drop(columns='number_of_missed_opportunities_during_lapsing_period')], ignore_index=True)
  df['is_missed_opportunity'] = False

  for i in range(max_nb_missed_opp):
    for a in range(i+1):
      missed_opp = flag_opportunities\
      .loc[flag_opportunities.number_of_missed_opportunities_during_lapsing_period == i+1].copy(deep=True)
      missed_opp['end_date'] = missed_opp['end_date'] + pd.to_timedelta(missed_opp['mean_cluster'] * (a + 1), unit='D')
      missed_opp.drop(columns='number_of_missed_opportunities_during_lapsing_period', inplace=True)
      missed_opp.opportunity_id.apply(lambda x: str(uuid.uuid4()), inplace=True)
      missed_opp['is_missed_opportunity'] = True
      df = pd.concat([missed_opp, df], ignore_index=True)

  df.drop(columns=['buying_frequency', 'buying_frequency_cluster', 'mean_cluster', 'last_opp_cluster_mean', 'change_to_slower_buying_frequency'], inplace=True)
  return df

df = addding_missed_opportunities_during_lapsing_period(df.copy(deep=True), flag_opportunities.copy(deep=True))

df = pd.concat([df, df_account_first_opp], ignore_index=True)

df_marketing_engagement = pd.read_csv('marketing_engagement.csv', usecols=['account_id', 'interaction_date'])
df_marketing_engagement['interaction_date'] = pd.to_datetime(df_marketing_engagement.interaction_date)

df.rename(columns={'end_date':'interaction_date'}, inplace=True)
df = pd.concat([df, df_marketing_engagement], ignore_index=True)

def mapping_the_influenced_flag(df):
  df.sort_values(['account_id', 'interaction_date'], inplace=True)
  df['opportunity_id'] = df.groupby('account_id').opportunity_id.apply(lambda x: x.bfill())
  df['is_missed_opportunity'] = df.groupby('account_id').is_missed_opportunity.apply(lambda x: x.bfill())
  df = df.loc[~df.is_missed_opportunity]
  influenced_flag = df.groupby('opportunity_id').interaction_date.transform(lambda x: 'influenced' if (len(x)>1) else 'non-influenced')
  influenced_flag.reset_index(inplace=True)
  influenced_flag.columns = ['opportunity_id', 'influenced_flag']
  return influenced_flag

influenced_flag = mapping_the_influenced_flag(df.copy(deep=True))

