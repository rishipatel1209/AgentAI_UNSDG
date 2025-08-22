sdg_goals={'poverty':'End poverty in all its forms everywhere','hunger':'Zero Hunger',\
           'health':'Good Health and Well-Being','education':'Quality Education',\
            'gender-equality':'Gender Equality','water-and-sanitation':'Clean Water and Sanitation',\
             'energy':'Affordable and Clean Energy','economic-growth':'Decent Work and Economic Growth',\
             'infrastructure-industrialization': 'Industries,Innovation, and Infrastructure',\
             'inequality':'Reduced Inequalities','cities':'Sustainable Cities and Communities',\
             'sustainable-consumption-production':'Responsible Consumption and Production',\
             'climate-change':'Climate Action','oceans':'Life Below Water','biodiversity':'Life on Land',\
             'peace-justice':'Peace, Justice, and Strong Instiutions', 'globalpartnerships':'Partnerships for the Goals'
}
sdg_topics={'poverty':['poverty-eradication'],
            'hunger':['rural-development','food-security-and-nutrition-and-sustainable-agriculture'],\
            'health':['health-and-population'],\
            'education':['education'],\
            'gender-equality':['gender-equality-and-womens-empowerment'],\
            'water-and-sanitation':['water-and-sanitation'],\
            'energy':['energy'],\
            'economic-growth':['employment-decent-work-all-and-social-protection'],\
            'infrastructure-industrialization':['industry'],\
            'inequality':[],\
            'cities':['sustainable-cities-and-human-settlements'],\
            'sustainable-consumption-production':[],\
            'climate-change':['green-economy'],\
            'oceans':['oceans-and-seas','small-island-developing-states'],\
            'biodiversity':['biodiversity-and-ecosystems','forests','mountains','desertification-land-degradation-and-drought'],\
            'peace-justice':['institutional-frameworks-and-international-cooperation-sustainable-development','violence-against-children'],\
            'globalpartnerships':[]}


# NewsAPI Supported Countries
# List of countries where NewsAPI sources their news from

newsapi_countries = ["United Arab Emirates",\
    "Argentina",\
    "Austria", \
    "Australia",\
    "Belgium",\
    "Bulgaria",\
    "Brazil",\
    "Canada",\
    "Switzerland",\
    "China",\
    "Colombia",\
    "Cuba",\
    "Czech Republic",\
    "Germany",\
    "Egypt",\
    "France",\
    "United Kingdom",\
    "Greece",\
    "Hong Kong",\
    "Hungary",\
    "Indonesia",\
    "Ireland",\
    "Israel",\
    "India",\
    "Italy",\
    "Japan",\
    "South Korea",\
    "Lithuania",\
    "Latvia",\
    "Morocco",\
    "Mexico",\
    "Malaysia",\
    "Nigeria",\
    "Netherlands",\
    "Norway",\
    "New Zealand",\
    "Philippines",\
    "Poland",\
    "Portugal",\
    "Romania",\
    "Serbia",\
    "Russia",\
    "Saudi Arabia",\
    "Sweden",\
    "Singapore",\
    "Slovenia",\
    "Slovakia",\
    "Thailand",\
    "Turkey",\
    "Taiwan",\
    "Ukraine",\
    "United States",\
    "Venezuela",\
    "South Africa"\
]
analytics_dict={'poverty':['share-in-poverty-relative-to-different-poverty-thresholds','consumer-price-index','intensity-of-multidimensional-poverty-hot']}
analytics_dict['hunger']=['daily-per-capita-caloric-supply','daily-protein-supply-from-animal-and-plant-based-foods','prevalence-of-undernourishment','prevalence-of-anemia-in-women-of-reproductive-age-aged-15-29','prevalence-of-anemia-in-children']
analytics_dict['health']=['life-expectancy','child-mortality','infectious-and-parasitic-diseases-death-rate-who-mdb','public-health-expenditure-share-gdp','maternal-mortality-slope-chart','global-vaccination-coverage']
analytics_dict['education']=['primary-enrollment-selected-countries','literacy-rate-of-young-men-and-women-line']
analytics_dict['gender-equality']=['gender-gap-education-levels','people-third-gender-recognized','key-features-of-womens-political-empowerment']
analytics_dict['water-and-sanitation']=['access-to-basic-services','drinking-water-services-coverage-rural','people-with-access-to-at-least-basic-drinking-water']
analytics_dict['energy']=['primary-sub-energy-source','electricity-fossil-renewables-nuclear-line','installed-global-renewable-energy-capacity-by-technology']
analytics_dict['economic-growth']=['global-incidence-of-child-labour-by-age-groups','unemployment-rate-imf','incidence-of-child-labour','annual-working-hours-per-worker']
analytics_dict['infrastructure-industrialization']=['broadband-penetration-by-country','ict-adoption-per-100-people','annual-research-and-development-funding-for-technologies-infectious-diseases']
analytics_dict['inequality']=['global-inequality-between-world-citizens-and-its-components','gini-coefficient-after-tax-lis']
analytics_dict['cities']=['urban-share-european-commission','population-density-of-the-capital-city','access-to-basic-services']
analytics_dict['sustainable-consumption-production']=['annual-co2-emissions-per-country','sulphur-dioxide-and-coal','treatment-of-hazardous-waste']
analytics_dict['climate-change']=['annual-share-of-co2-emissions','contributions-global-temp-change','long-run-air-pollution','change-air-pollutant-emissions','electronic-waste-recycling-rate']
analytics_dict['oceans']=['wild-caught-fish','aquaculture-farmed-fish-production','industrial-water-as-a-share-of-total-water-withdrawals']
analytics_dict['biodiversity']=['total-agricultural-land-use-per-person','land-use-agriculture-longterm','land-natural-share','area-of-permanent-meadows-and-pastures']
analytics_dict['peace-justice']=['democracy-index-by-source','key-media-freedoms','key-features-of-democracy','key-features-of-womens-political-empowerment','deaths-in-armed-conflicts','deaths-in-state-based-conflicts','terrorist-attacks-fatalities-and-injuries']