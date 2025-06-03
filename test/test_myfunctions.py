#from mypythonlib import myfunctions

#remote_get_classifier_user_details.py

# Polarization code listing can be found in:
# C:\Users\tja12\Dropbox (Sydney Uni)\Projects\Opinion_spread\Twitter_code\my_Twitter_library_output

# Plot median types and tokens (per user) by data set, with text labels inside circles (for left and right quintiles)

in_dir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
out_dir = in_dir
flags = ['trump_23_06_2018','trump_realDonaldTrump_potus_18_09_2018','kavanaugh_scotus_20_09_2018','election_midterm_09_11_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020','biden_03_11_2021']

get_stats_hashtag_types_tokens(in_dir,out_dir,flags)

# Plot as scatter plot all users (within flag) tokens vs types (per user), for left and right quintiles (note biden quintiles are reversed)

in_dir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
flag = 'biden_03_11_2021'
#flags = ['trump_23_06_2018','trump_realDonaldTrump_potus_18_09_2018','kavanaugh_scotus_20_09_2018','election_midterm_09_11_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020','biden_03_11_2021']

plot_hashtag_types_tokens(in_dir, flag)


# Determine tokens and types from hashtag files

in_dir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
out_dir = in_dir
flags = ['trump_23_06_2018','trump_realDonaldTrump_potus_18_09_2018','kavanaugh_scotus_20_09_2018','election_midterm_09_11_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020','biden_03_11_2021']

max_items = 2000

get_user_hashtag_types_tokens(in_dir,out_dir,flags,max_items)


# User distribution figure (for figure 3)
in_dir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
out_dir = in_dir
flags = ['trump_23_06_2018','trump_realDonaldTrump_potus_18_09_2018','kavanaugh_scotus_20_09_2018','election_midterm_09_11_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020','biden_03_11_2021']

max_items = 2000
 
fig3_plot_distributions(in_dir,out_dir,flags,max_items,21)


#Topic figure (for figure 2)
indir = 'C:/Users/tja12/Dropbox (Sydney Uni)/Projects/Opinion_spread/Text/Polarization/Figures/Data/'
outdir = indir
filebase = 'trump_realDonaldTrump_04_09_2020'
plot_topics(indir,outdir, filebase)


# New topic modelling approach (using lda_mallet)
indir = 'C:/Users/tja12/Dropbox (Sydney Uni)/Projects/Opinion_spread/Text/Polarization/Figures/Data/'
outdir = indir+'Check/'
filebase = 'trump_realDonaldTrump_04_09_2020'
outflag = '_seed_123'
classbase = 'trump_realDonaldTrump_04_09_2020'
max_items = 2000
direct = True
ntopics = 14
user_description_topic_modelling(indir,outdir,classbase,filebase,outflag,max_items,direct,ntopics)

#indir = 'C:/Users/tja12/Dropbox (Sydney Uni)/Projects/Opinion_spread/Text/Polarization/Figures/Data/'
#outdir = indir
#filebase = 'trump_realDonaldTrump_04_09_2020'
#max_items = 2000
#ntopics = 14
#user_description_topic_modelling(indir,outdir,filebase,max_items,ntopics)


#Old topic modelling approach (from jupyter notebook)
#in_dir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
in_dir = 'C:/Users/tja12/Dropbox (Sydney Uni)/Projects/Opinion_spread/Text/Polarization/Figures/Data/'
out_dir = in_dir

classbase = 'trump_realDonaldTrump_04_09_2020'
max_items = 2000
ntopics = 3
#get_user_text(in_dir,out_dir, classbase)
user_description_topic_modellingv2(in_dir,classbase,max_items,ntopics)


# Process out_user files to extract only username and text description

in_dir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
out_dir = in_dir

classbase = 'trump_realDonaldTrump_04_09_2020'
get_user_text(in_dir,out_dir, classbase)



# Plot congress check figure

indir = 'C:/Users/tja12/Dropbox (Sydney Uni)/Projects/Opinion_spread/Text/Polarization/Figures/Data/'
outdir = indir
filebase = '116Congress'
max_items = 2000
fig2_plot_congress(indir,outdir, filebase, max_items)


# Classify congress members using retweet classifications given in alternate file

in_dir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
out_dir = in_dir

classbase = 'trump_realDonaldTrump_04_09_2020'
filebase = '116Congress'
max_items = 2000

classify_congress(in_dir,out_dir,classbase,filebase,max_items)


# Processing overlap matrices, then classifying users

in_dir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
out_dir = in_dir
flags = ['trump_23_06_2018','trump_realDonaldTrump_potus_18_09_2018','kavanaugh_scotus_20_09_2018','election_midterm_09_11_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020','biden_03_11_2021']

max_items = 2000

for filebase in flags:

    pca_calc(in_dir,out_dir,filebase,max_items)
    remote_classify_all_select_no_lang_check(in_dir, out_dir, filebase, max_items)


# Plot of retweet/hashtag data (using biden data set) for Fig. 1
in_dir = 'C:/Users/tja12/Dropbox (Sydney Uni)/Projects/Opinion_spread/Text/Polarization/Figures/Data/'
out_dir = in_dir
max_items = 2000
filebase = 'biden_03_11_2021'
max_display = 21

fig1_plot_hash_retweet(in_dir, out_dir, filebase, max_items, max_display)

# Plot schematic between users and retweeted users

in_dir = '/import/silo4/tale6898/Twitter_data/'
out_dir = in_dir
filebase = 'biden_03_11_2021'
num_retweeted = 10
num_users = 20
fig1_plot_schem_graph(in_dir,out_dir,filebase,num_retweeted,num_users)


# Get edges between users and most retweeted users

in_dir = '/import/silo4/tale6898/Twitter_data/'
out_dir = in_dir
filebase = 'biden_03_11_2021'
max_items = 2000
num_retweeted = 10
num_users = 20
get_user_edges(in_dir,out_dir,filebase,max_items,num_retweeted,num_users)


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


# Calculate user pressure

in_dir = '/import/silo4/tale6898/Twitter_data/'
out_dir = '/import/silo4/tale6898/Twitter_data/'
flags = ['trump_23_06_2018','election_midterm_09_11_2018','kavanaugh_scotus_20_09_2018','trump_realDonaldTrump_potus_18_09_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020']
date_time = '02_11_2021'
file_flag = 'US_Politics'
for flag in flags:
    get_user_pressure(in_dir,out_dir,flags,date_time,file_flag)


# Process down-sampled outputs (i.e. from classified users active at the same time).  All retweet stats provided by user
# Output: base_dir + fflag + '_sample_out_all_retweet_interactions.csv'

sample_dir = '/suphys/tale6898/Documents/Twitter_data/Processed/Samples/'
base_dir = '/import/silo4/tale6898/Twitter_data/'
flags = ['trump_23_06_2018','election_midterm_09_11_2018','kavanaugh_scotus_20_09_2018','trump_realDonaldTrump_potus_18_09_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020']
#flags = ['trump_23_06_2018','election_midterm_09_11_2018','trump_realDonaldTrump_potus_18_09_2018','check_trump_23_06_2018_15_11_2019']
for flag in flags:
    get_retweet_stats(sample_dir,base_dir,flag)


# Check fraction of reply in sample accounts that survived
in_dir = '/import/silo4/tale6898/Twitter_data/'
flags = ['trump_23_06_2018','trump_realDonaldTrump_potus_18_09_2018','kavanaugh_scotus_20_09_2018','election_midterm_09_11_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020']
out_dir = '/import/silo4/tale6898/Twitter_data/'
date_time = '02_11_2021'
flag = 'US_Politics'
get_subset_survival_frac(in_dir,out_dir,flags,date_time,flag)


# Check existence of user accounts
in_dir = '/import/silo4/tale6898/Twitter_data/'
out_dir = '/import/silo4/tale6898/Twitter_data/'
flag = 'US_Politics'
check_subset_users(in_dir, out_dir, flag)


# Obtain ratios of replies between communities, and activity levels of communities over the sampled period of interest (follows get_reply_stats)

base_dir = '/import/silo4/tale6898/Twitter_data/'
flags = ['trump_23_06_2018','trump_realDonaldTrump_potus_18_09_2018','kavanaugh_scotus_20_09_2018','election_midterm_09_11_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020']
process_reply_stats(base_dir,flags)


# Process down-sampled outputs (i.e. from classified users active at the same time).  All reply stats provided by user
# Output: base_dir + fflag + '_sample_out_all_reply_interactions.csv'

sample_dir = '/suphys/tale6898/Documents/Twitter_data/Processed/Samples/'
base_dir = '/import/silo4/tale6898/Twitter_data/'
#flags = ['trump_23_06_2018','election_midterm_09_11_2018','kavanaugh_scotus_20_09_2018','trump_realDonaldTrump_potus_18_09_2018','realDonaldTrump_trump_18_04_2019','check_trump_23_06_2018_15_11_2019','trump_realDonaldTrump_04_09_2020']
flags = ['trump_23_06_2018','election_midterm_09_11_2018','trump_realDonaldTrump_potus_18_09_2018','check_trump_23_06_2018_15_11_2019']
for flag in flags:
    get_reply_stats(sample_dir,base_dir,flag)




# Extract retweets, replies and hashtags to files by user, and produce frequency files of these characteristics

in_dir = '/import/silo4/tale6898/Twitter_data/Aus_climate/All_users/'
out_dir = '/import/silo4/tale6898/Twitter_data/Aus_climate/Processed_users/'
filebase = 'Aus_climate'

extract_to_file_hashtag_retweet_reply(filebase,in_dir,out_dir)

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

# Calculate overlap matrix based on retweet activity (takes around 18 hours for 134000 climate users)

in_dir = '/import/silo4/tale6898/Twitter_data/Aus_climate/Processed_users/'
filebase = 'Aus_climate'
max_items = 2000
filefreq = in_dir+'retweeted_freq_'+filebase+'.csv'
filein = in_dir+filebase+'_out_retweet.txt'
fileout = in_dir+filebase+'_'+str(max_items)+'_retweet_overlap.txt'
hash = 0

overlap_calc(filefreq,filein,fileout,hash,max_items)

# Classify top 2000 retweeted accounts

in_dir = '/import/silo4/tale6898/Twitter_data/Aus_climate/Processed_users/'
out_dir = '/import/silo4/tale6898/Twitter_data/Aus_climate/Processed_users/'
filebase = 'Aus_climate'
max_items = 2000
pca_calc(in_dir,out_dir,filebase,max_items)


#Working implementation for extract_user_details() and get_user_samples()

filebase = ['trump_23_06_2018','election_midterm_09_11_2018','kavanaugh_scotus_20_09_2018','trump_realDonaldTrump_potus_18_09_2018','realDonaldTrump_trump_18_04_2019','check_trump_23_06_2018_15_11_2019','trump_realDonaldTrump_04_09_2020']

file_dir = '/suphys/tale6898/Documents/Twitter_data/Processed/'
out_dir = '/suphys/tale6898/Documents/Twitter_data/Processed/Samples/'

for fflag in filebase:
    extract_user_details(fflag,file_dir)
    get_user_samples(fflag,file_dir,out_dir,14)


# Code needed to run down_sample_users()

filebase = ['trump_23_06_2018','election_midterm_09_11_2018','kavanaugh_scotus_20_09_2018','trump_realDonaldTrump_potus_18_09_2018','realDonaldTrump_trump_18_04_2019','check_trump_23_06_2018_15_11_2019','trump_realDonaldTrump_04_09_2020']
sample_dir = '/suphys/tale6898/Documents/Twitter_data/Processed/Samples/'
base_dir = '/import/silo4/tale6898/Twitter_data/'

for fflag in filebase:


    down_sample_users(sample_dir,base_dir,fflag)

# Code needed to downsample further to only non-verified captured users, and then determine all replies in to that user

filebase = ['trump_23_06_2018','election_midterm_09_11_2018','kavanaugh_scotus_20_09_2018','trump_realDonaldTrump_potus_18_09_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020','check_trump_23_06_2018_15_11_2019']
sample_dir = '/suphys/tale6898/Documents/Twitter_data/Processed/Samples/'
base_dir = '/import/silo4/tale6898/Twitter_data/'

for fflag in filebase:

    select_reply_sample(sample_dir,base_dir,fflag)
    get_in_replies(sample_dir,base_dir,fflag)

# Code needed to get reply statistics and summary

filebase = ['election_midterm_09_11_2018','kavanaugh_scotus_20_09_2018']
sample_dir = '/suphys/tale6898/Documents/Twitter_data/Processed/Samples/'
base_dir = '/import/silo4/tale6898/Twitter_data/'

for fflag in filebase:

    get_reply_ratio(sample_dir,base_dir,fflag)


# Code needed to get reply and retweet statistics to @realDonaldTrump, by left and right user

trump_user_filebase = ['trump_23_06_2018','election_midterm_09_11_2018','kavanaugh_scotus_20_09_2018','trump_realDonaldTrump_potus_18_09_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020']

sample_dir = '/suphys/tale6898/Documents/Twitter_data/Processed/Samples/'
base_dir = '/import/silo4/tale6898/Twitter_data/'
all_sample_data = list()
all_prob_data = list()
target = 'realDonaldTrump'
for fflag in trump_user_filebase:

    [sample_data,prob_data] = get_probs(sample_dir,base_dir,fflag,target)
    all_sample_data.append(sample_data)
    all_prob_data.append(prob_data)

with open(base_dir+target+'_all_stream_sample_activity_totals.txt', 'w') as f:
    for sample_data in all_sample_data:
        f.write(str(sample_data))
        f.write('\n')

with open(base_dir+target+'_all_stream_sample_prob_right_retweet.txt', 'w') as f:
    for prob_data in all_prob_data:
        f.write(str(prob_data[0:4]))
        f.write('\n')

with open(base_dir+target+'_all_stream_sample_prob_left_reply.txt', 'w') as f:
    for prob_data in all_prob_data:
        f.write(str(prob_data[4:8]))
        f.write('\n')


# Code needed to extract all tweets, replies, quotes and retweets from climate streams.  Data is aggregated by day
# Output takes form [date,user_num,retweeted/replied/quoted user num, original tweet id]

climate_stream_filebase = ['climate_12_11_2019','climate_14_11_2019','climate_12_12_2019','climate_20_01_2020','climate_13_02_2020','climate_13_03_2020','climate_16_03_2020','climate_17_04_2020','climate_16_05_2020','climate_17_06_2020','climate_03_07_2020','climate_11_07_2020','climate_18_08_2020','climate_01_09_2020','climate_04_09_2020','climate_10_09_2020','climate_26_09_2020','climate_08_10_2020','climate_15_12_2020','climate_20_01_2021','climate_24_01_2021','climate_14_02_2021','climate_17_02_2021','climate_20_02_2021','climate_23_02_2021','climate_24_02_2021','climate_26_02_2021','climate_01_03_2021','climate_04_03_2021','climate_08_03_2021','climate_13_04_2021','climate_14_04_2021','climate_20_04_2021','climate_25_06_2021','climate_02_07_2021','climate_17_07_2021']
in_dir = '/import/silo4/tale6898/Twitter_data/'
out_dir = '/import/silo4/tale6898/Twitter_data/Climate_cleaned/'
base_flag = 'climate'
down_sample_streams(in_dir,out_dir,climate_stream_filebase,base_flag)


# Code needed to get classifier details from users and then plot verified vs non-verified distribution

max_items = 2000

flaglist = ['trump_23_06_2018','trump_realDonaldTrump_potus_18_09_2018','kavanaugh_scotus_20_09_2018','election_midterm_09_11_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020', 'biden_03_11_2021']

dirbase = '/import/silo4/tale6898/Twitter_data/users_'
classdir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
outdir = classdir
indir = classdir
nbins = 5
# Have to be careful with the following, as the stem for users changes, current format only works for most recent collections
#get_classifier_details(outdir, dirbase, classdir, flaglist, max_items)

make_classifier_figures(indir,flaglist,nbins)


# Topic modelling figure for user descriptors in polarization paper

indir = 'C:/Users/tja12/Dropbox (Sydney Uni)/Projects/Opinion_spread/Text/Polarization/Figures/Data/'
outdir = indir+'Check/'
filebase = 'trump_realDonaldTrump_04_09_2020'
rseed = 128
classbase = 'trump_realDonaldTrump_04_09_2020'
max_items = 2000
direct = True
ntopics = 14

user_description_topic_modelling(indir,outdir,classbase,filebase,rseed,max_items,direct,ntopics)

descriptors = ['Political left','Occupation','Interest','Political','Professional','News','Life','Twitter','Male','Activism','Block','Proud','Love','Political right']

plot_topics(indir, outdir, filebase, rseed, descriptors,ntopics)


# Get classifier (i.e. top 2000 retweeted) details

max_items = 2000

flaglist = ['trump_23_06_2018','trump_realDonaldTrump_potus_18_09_2018','kavanaugh_scotus_20_09_2018','election_midterm_09_11_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020', 'biden_03_11_2021']

dirbase = '/import/silo4/tale6898/Twitter_data/users_'
classdir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
outdir = classdir
indir = classdir
nbins = 5
get_classifier_details(outdir, dirbase, classdir, flaglist, max_items)

# Get classifier descriptions (potentially for further analysis i.e. classification or topic modelling)

get_classifier_text(indir,outdir,flaglist)

# Check if classifier accounts are still present

fflag = 'total_classifier_set_US_Politics'

classdir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
outdir = classdir
indir = classdir

check_classifier_users(indir,outdir,fflag)

# get list of all classifiers, for 'existfile' below

get_all_classifiers(indir, outdir, flaglist)

# Needs information copied to the below directory

indir = 'C:/Users/tja12/Dropbox (Sydney Uni)/Projects/Opinion_spread/Text/Polarization/Figures/Data/'
outdir = indir

existfile = 'total_classifier_set_US_Politics_existence_16_07_2022.csv'

make_classifier_figures(indir,flaglist,nbins,existfile)



indir = '/import/silo4/tale6898/Twitter_data/Processed_political_users/'
outdir = indir
userdir = '/import/silo4/tale6898/Twitter_data/users_'
rmax = 2000000
ncrit = 10

flaglist = ['trump_23_06_2018','trump_realDonaldTrump_potus_18_09_2018','kavanaugh_scotus_20_09_2018','election_midterm_09_11_2018','realDonaldTrump_trump_18_04_2019','trump_realDonaldTrump_04_09_2020', 'biden_03_11_2021']

for fileflag in flaglist:
    print(fileflag)
    # gets top retweets of verified accounts
    extract_to_file_top_retweets(indir,userdir,outdir,fileflag,rmax,ncrit)
    # gets activity of users with respect to all classifier accounts
    extract_retweet_reply_to_verified(indir,userdir,outdir,fileflag)

