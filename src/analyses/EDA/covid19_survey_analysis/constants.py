# Glimpse of all the features present in the questionare of the survey

village_info_col = ["Block - Panchayat - Volunteer - Block",
                    "Block - Panchayat - Volunteer - Gram Panchayat",
                    "How many villages are there in your panchayat?",
                    "Have you maintained a record of the Migrant Community who returned during lockdown?",
                    "Name of Village",
                    ]

questionare_col = [
    "Tick the schemes that you are aware of",
    "What facilities are available for returned Migrants in your Panchayat?",
    "What activities have been initiated in your Panchayat for safety from COVID-19?",
    "What are the reasons?",
    "Are these community committees there in your Panchayat?",
    "What activities are going on in your panchayat to provide livelihoods to returned migrant community?",
    "Under MNREGA, what activities are going on in your panchayat to provide a livelihood to the returned migrant community?",
    "What kind of Collaboration is desired?",
]

other_ques_col = [
    "Tick the schemes that you are aware of--OTHER--",
    "What facilities are available for returned Migrants in your Panchayat?--OTHER--",
    "What activities have been initiated in your Panchayat for safety from COVID-19?--OTHER--",
    "What are the reasons?--OTHER--",
    "Are these community committees there in your Panchayat?--OTHER--",
    "What activities are going on in your panchayat to provide livelihoods to returned migrant community?--OTHER--",
    "Under MNREGA, what activities are going on in your panchayat to provide a livelihood to the returned migrant community?--OTHER--",
    "What kind of Collaboration is desired?--OTHER--",
]

# Options of the Optional Question
option_ques = ["--OPTION--|Pradhanmantri Kisan Samman Yojana",
               "--OPTION--|Pradhanmantri Jan-Dhan Khata yojana",
               "--OPTION--|Pradhanmantri Gareen Ann Kalyan Yojana",
               "--OPTION--|Pradhanmantri Ujjwala Yojana",
               "--OPTION--|Rashtriya Samajik Sahayta Karyakram",
               "--OPTION--|Samajik Suraksha Pension",
               "--OPTION--|Pradhanmantri Gareeb Kalyan Rozgar Abhiyan Yojana",
               "--OPTION--|Deendayal Yojana for Self Help Groups",
               "--OPTION--|Help to Special Backward Tribes",
               "--OPTION--|Ready to use food packet distribution in Anganwadi centres",
               "--OPTION--|Mid-Day Meal Scheme",
               "--OPTION--|Mazdoor Sahayta Yojana for registered Construction Labour",
               "--OPTION--|Madhya Pradesh Rozgar Setu Yojana",
               "--OPTION--|Pradhanmantri Swanidhi Yojana Nidhi",
               "--OPTION--|Free treatment of COVID-19",
               "--OPTION--|MGNREGA",
               "--OPTION--|Don't have any information",
               "--OPTION--|Quarantine Centre",
               "--OPTION--|Ration Distribution Centre",
               "--OPTION--|MGNREGA.1",
               "--OPTION--|Anganwadi Centre",
               "--OPTION--|Mid-Day Meal",
               "--OPTION--|Health Centre",
               "--OPTION--|Other livelihood resources",
               "--OPTION--|Mask distribution",
               "--OPTION--|Circle making for social distancing",
               "--OPTION--|Soap/Sanitizer distribution",
               "--OPTION--|Ration kit distribution",
               "--OPTION--|Screening/Identification of primary symptoms",
               "--OPTION--|Wall Painting",
               "--OPTION--|None",
               "--OPTION--|Farming Production Centre",
               "--OPTION--|Mahila Mandal",
               "--OPTION--|Farmers Group",
               "--OPTION--|Self Help Groups",
               "--OPTION--|Disaster Management Committee",
               "--OPTION--|Yuva Mandal",
               "--OPTION--|Jan Abhiyan Parishad Samiti (Prasphutan Samiti)",
               "--OPTION--|Soap Production",
               "--OPTION--|Mask Production",
               "--OPTION--|Sanitizer Production",
               "--OPTION--|Take-Home Ration",
               "--OPTION--|Group making",
               "--OPTION--|Repair and maintenance of Ponds",
               "--OPTION--|Water shed",
               "--OPTION--|Small enterprises for agriculture-related work",
               "--OPTION--|Help for an awareness campaign",
               "--OPTION--|Skill development training",
               "--OPTION--|Seeds and Guidance for Nutrition Garden",
               "--OPTION--|Schemes related information and help",
               ]

categorical_ques = [
    "How many villages are there in your panchayat?",
    "Have you maintained a record of the Migrant Community who returned during lockdown?",
    "Total number of Migrant Families who have returned",
    "Females",
    "Males",
    "Children (0-6 years)",
    "Total Number of COVID Infected",
    "How many COVID-19 testing centres or treatment centres are there in your Panchayat?",
    "Are you aware of various Government Schemes for Migrant Labours who have returned?",
    "Are milk, vegetables and daily use essentials supply easily available in your Panchayat?",
    "We, under COVID-19 response program, will work for livelihood support to returned migrant labour in your panchayat, in which your support is also required. Do you agree?",
]


# Non categorical features used for regression of target variable
regression_feat = ["How many villages are there in your panchayat?",
                   "Total number of Migrant Families who have returned",
                   "Females",
                   "Males",
                   "Children (0-6 years)",
                   "Total Number of COVID Infected",
                   "How many COVID-19 testing centres or treatment centres are there in your Panchayat?",
                   ]


# Categorical features used for regression of target variable
categorical_feat = option_ques + [
    "Have you maintained a record of the Migrant Community who returned during lockdown?",
    "Are you aware of various Government Schemes for Migrant Labours who have returned?",
    "Are milk, vegetables and daily use essentials supply easily available in your Panchayat?",
    "We, under COVID-19 response program, will work for livelihood support to returned migrant labour in your panchayat, in which your support is also required. Do you agree?",
]

# Index columns of the dataframe
index_col = "Identifier"

# Target column which need to be regressed
target_col = 'Total Number of COVID Infected'
