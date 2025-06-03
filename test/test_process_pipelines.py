#from mypythonlib import myfunctions

# Full process pipeline

in_dir = '/import/silo4/tale6898/Twitter_data/users_biden_03_11_2021/'
out_dir = '/import/silo4/tale6898/Twitter_data/'
filebase = 'biden_03_11_2021'


in_dir = out_dir

max_items = 2000
filefreq = in_dir+'hashtags_freq_'+filebase+'.csv'
filein = in_dir+filebase+'_out_hash.txt'
fileout = in_dir+filebase+'_'+str(max_items)+'_hashtag_overlap.txt'
#filefreq = in_dir+'retweeted_freq_'+filebase+'.csv'
#filein = in_dir+filebase+'_out_retweet.txt'
#fileout = in_dir+filebase+'_'+str(max_items)+'_retweet_overlap.txt'
hash = 1

overlap_calc(filefreq,filein,fileout,hash,max_items)
#pca_calc(in_dir,out_dir,filebase,max_items)
#remote_classify_all_select_no_lang_check(in_dir, out_dir, filebase, max_items)
#plot_user_distribution(in_dir, out_dir, filebase, max_items)


# 1. From user jsonl files, extract retweets, replies and hashtags

in_dir = '/import/silo4/tale6898/Twitter_data/Aus_climate/All_users/'
out_dir = '/import/silo4/tale6898/Twitter_data/Aus_climate/Processed_users/'
filebase = 'Aus_climate'

extract_to_file_hashtag_retweet_reply(filebase,in_dir,out_dir)

# 2. From processed user files, extract rank-frequency list of hashtags, retweets, replies

in_dir = out_dir

base_file = in_dir+filebase+'_out_hash.txt'
hash = 1
fileout = out_dir+'hashtags_freq_'+filebase+'.csv'
process_all(base_file,fileout,hash)

base_file = in_dir+filebase+'_out_retweet.txt'
hash = 0
fileout = out_dir+'retweeted_freq_'+filebase+'.csv'
process_all(base_file,fileout,hash)

base_file = in_dir+filebase+'_out_reply.txt'
hash = 0
fileout = out_dir+'replied_users_freq_'+filebase+'.csv'
process_all(base_file,fileout,hash)

# 3. Calculate overlap matrix.  Goes through all user out lists and calculates overlap (could be optimized)
# Calculate overlap matrix based on retweet activity (takes around 18 hours for 134000 climate users)

in_dir = '/import/silo4/tale6898/Twitter_data/Aus_climate/Processed_users/'
filebase = 'Aus_climate'
max_items = 2000
filefreq = in_dir+'retweeted_freq_'+filebase+'.csv'
filein = in_dir+filebase+'_out_retweet.txt'
fileout = in_dir+filebase+'_'+str(max_items)+'_retweet_overlap.txt'
hash = 0

overlap_calc(filefreq,filein,fileout,hash,max_items)

# 4. Project top 2000 retweeted accounts on to low-dimensional space.  Currently doing this without scaling.
# Mixed results when scaling.  Seems to work well for political accounts, but classification doesn't make sense for climate set
# E.g. malcolmroberts and GretaThunberg have similar first dimensions after scaling which is strange
# Classify top 2000 retweeted accounts

in_dir = '/import/silo4/tale6898/Twitter_data/Aus_climate/Processed_users/'
out_dir = '/import/silo4/tale6898/Twitter_data/Aus_climate/Processed_users/'
filebase = 'Aus_climate'
max_items = 2000
pca_calc(in_dir,out_dir,filebase,max_items)

# 5. Classify all users according to retweet activity

in_dir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
out_dir = in_dir
flags = ['trump_23_06_2018','trump_realDonaldTrump_potus_18_09_2018','kavanaugh_scotus_20_09_2018','election_midterm_09_11_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020','biden_03_11_2021']

max_items = 2000

for filebase in flags:

#    pca_calc(in_dir,out_dir,filebase,max_items)
    remote_classify_all_select_no_lang_check(in_dir, out_dir, filebase, max_items)

# 6. Schematic figure of top 20 users and top 10 most retweeted accounts (for fig. 1)

in_dir = '/import/silo4/tale6898/Twitter_data/'
out_dir = in_dir
filebase = 'biden_03_11_2021'
max_items = 2000
num_retweeted = 10
num_users = 20
get_user_edges(in_dir,out_dir,filebase,max_items,num_retweeted,num_users)
fig1_plot_schem_graph(in_dir,out_dir,filebase,num_retweeted,num_users)

# 7. Plot of retweet/hashtag data (using biden data set) for Fig. 1

in_dir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
#in_dir = 'C:/Users/tja12/Dropbox (Sydney Uni)/Projects/Opinion_spread/Text/Polarization/Figures/Data/'
out_dir = in_dir
max_items = 2000
filebase = 'biden_03_11_2021'
max_display = 21

fig1_plot_hash_retweet(in_dir, out_dir, filebase, max_items, max_display)

# 8. Create Congress classification figure

in_dir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
out_dir = in_dir

classbase = 'trump_realDonaldTrump_04_09_2020'
filebase = '116Congress'
max_items = 2000

classify_congress(in_dir,out_dir,classbase,filebase,max_items)
fig2_plot_congress(in_dir,out_dir, filebase, max_items)

# 9. Topic modelling of user descriptions (still needs further work, particularly check of projection of users back onto dominant topics)

indir = 'C:/Users/tja12/Dropbox (Sydney Uni)/Projects/Opinion_spread/Text/Polarization/Figures/Data/'
outdir = indir
filebase = 'trump_realDonaldTrump_04_09_2020'
outflag = '_new'
classbase = 'trump_realDonaldTrump_04_09_2020'
max_items = 2000
direct = True
ntopics = 10
#ntopics = 14
user_description_topic_modelling(indir,outdir,classbase,filebase,outflag,max_items,direct,ntopics)




