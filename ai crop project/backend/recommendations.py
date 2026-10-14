"""
PlantCare AI - Disease Knowledge Base and Recommendations Engine
Provides structured agronomic insights, symptoms, contributing factors,
immediate care protocols, and long-term prevention methods.
"""

from typing import Dict, Any, List

# Structured knowledge base for supported plant disease classes
DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Apple___Apple_scab": {
        "plant": "Apple",
        "disease": "Apple Scab (Venturia inaequalis)",
        "is_healthy": False,
        "severity": "Moderate to High",
        "description": "A prevalent fungal disease attacking Malus species that causes dark, olive-green to black scabby lesions on leaves and fruit, leading to defoliation and stunted fruit growth.",
        "symptoms": [
            "Olive-green to velvety dark brown lesions on the upper leaf surface.",
            "Leaves turn yellow, curl, and drop prematurely in late spring/early summer.",
            "Rough, corky brown scabs developing on developing fruit."
        ],
        "causes": [
            "Overwintering fungal spores in fallen leaf litter.",
            "Prolonged leaf wetness (6-9+ hours) during cool spring temperatures (55-75°F / 13-24°C).",
            "Dense canopy limiting air movement and sunlight penetration."
        ],
        "immediate_actions": [
            "Rake up and destroy fallen, infected leaf litter around the tree base.",
            "Prune out dead or overlapping branches during dry weather to maximize airflow.",
            "Avoid overhead irrigation; water the root zone directly."
        ],
        "preventative_measures": [
            "Plant resistant apple cultivars (e.g., Liberty, Enterprise, Freedom, Honeycrisp).",
            "Apply preventative sulfur or copper-based horticultural sprays early in the season at green-tip stage.",
            "Maintain an open canopy structure through annual dormant pruning."
        ],
        "organic_controls": [
            "Neem oil or horticultural oil sprays applied as preventative measures.",
            "Potassium bicarbonate sprays at first sign of disease outbreak."
        ],
        "expert_consultation": "Contact your local agricultural extension service if more than 30% of the canopy shows defoliation or if commercial orchard yields are threatened."
    },
    "Apple___Black_rot": {
        "plant": "Apple",
        "disease": "Black Rot (Botryosphaeria obtusa)",
        "is_healthy": False,
        "severity": "High",
        "description": "A destructive fungal pathogen causing 'frog-eye' leaf spots, cankers on branches, and black, mummified rotting fruit.",
        "symptoms": [
            "Small purple specks on leaves that enlarge into circular spots with tan centers ('frog-eye' appearance).",
            "Sunken reddish-brown cankers on twigs and limbs.",
            "Fruit rots, turns completely black, and shrivels into a hard mummy."
        ],
        "causes": [
            "Warm, humid conditions (75-85°F / 24-29°C) with persistent moisture.",
            "Infection entering through bark wounds, insect damage, or fire blight strikes.",
            "Presence of dead wood and mummified fruit left on trees from previous seasons."
        ],
        "immediate_actions": [
            "Remove and incinerate all mummified fruit hanging on the tree or on the ground.",
            "Prune back infected twigs and branches at least 6-8 inches below visible cankers with sterilized tools.",
            "Sterilize pruning shears between cuts using 70% isopropyl alcohol."
        ],
        "preventative_measures": [
            "Keep trees vigorous through balanced fertilization (avoid excess nitrogen).",
            "Protect trees against insect wounding (borers, codling moth).",
            "Apply captan or copper fungicide sprays beginning at tight cluster stage according to label instructions."
        ],
        "organic_controls": [
            "Bio-fungicides containing Bacillus subtilis or liquid copper formulations applied prior to rain events."
        ],
        "expert_consultation": "Consult a certified arborist or agricultural agent if trunk cankers threaten tree structural integrity."
    },
    "Apple___Cedar_apple_rust": {
        "plant": "Apple",
        "disease": "Cedar Apple Rust (Gymnosporangium juniperi-virginianae)",
        "is_healthy": False,
        "severity": "Moderate",
        "description": "A heteroecious fungal rust requiring both apple/crabapple trees and eastern red cedar/juniper species to complete its complex two-year lifecycle.",
        "symptoms": [
            "Bright yellow-orange or reddish-orange circular spots on upper leaf surfaces.",
            "Small black dots (pycnia) inside the orange spots.",
            "Fringed, tube-like projections (aecia) on the underside of leaves and fruit in late summer."
        ],
        "causes": [
            "Windblown spores from nearby Eastern Red Cedar or Juniper galls during spring rains.",
            "Wet conditions with temperatures around 55-75°F (13-24°C)."
        ],
        "immediate_actions": [
            "Inspect neighboring juniper or cedar trees within 1-2 miles and remove visible gelatinous orange galls if feasible.",
            "Remove severely rusted leaves from young saplings to preserve photosynthetic capacity."
        ],
        "preventative_measures": [
            "Select rust-immune apple cultivars (e.g., Redfree, Priscilla, Williams Pride).",
            "Avoid planting ornamental junipers within close proximity (500+ yards) to orchards.",
            "Apply protective fungicide sprays (myclobutanil or sulfur) from pink bud stage through petal fall."
        ],
        "organic_controls": [
            "Sulfur-based fungicides applied preventatively before rain events in early spring."
        ],
        "expert_consultation": "Seek extension advice if defoliation exceeds 25% for two consecutive seasons."
    },
    "Apple___healthy": {
        "plant": "Apple",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "description": "The foliage displays vibrant green pigmentation, intact margins, and no visible signs of fungal, bacterial, or insect damage.",
        "symptoms": [
            "Uniform green leaf blade without spotting, necrosis, or chlorosis.",
            "Clean leaf margins without curling or chewing damage.",
            "Normal leaf thickness and active photosynthesis."
        ],
        "causes": [
            "Optimal growing conditions, balanced nutrients, and effective orchard sanitation."
        ],
        "immediate_actions": [
            "No immediate treatment required.",
            "Maintain regular orchard scouting and routine watering schedules."
        ],
        "preventative_measures": [
            "Maintain deep root watering during dry spells (1 inch of water per week).",
            "Apply mulch around the drip line (keeping 4 inches away from trunk bark).",
            "Monitor weekly for early signs of aphids, mites, or leaf spot pathogens."
        ],
        "organic_controls": [
            "Compost tea sprays to foster beneficial microbial phyllosphere populations."
        ],
        "expert_consultation": "Routine extension soil testing every 2-3 years to ensure optimal soil pH (6.0-6.8)."
    },
    "Corn_(maize)___Common_rust_": {
        "plant": "Corn (Maize)",
        "disease": "Common Rust (Puccinia sorghi)",
        "is_healthy": False,
        "severity": "Moderate",
        "description": "A wind-dispersed fungal rust affecting sweet corn and field corn, producing powdery golden-brown pustules on both leaf surfaces.",
        "symptoms": [
            "Oval to elongate cinnamon-brown pustules (uredinia) scattered across both leaf surfaces.",
            "Pustules rupture the epidermis, releasing powdery rust-colored spores when rubbed.",
            "Leaves turn chlorotic and dry out when pustules coalesce heavily."
        ],
        "causes": [
            "Spores transported from southern regions on storm fronts.",
            "High relative humidity (>95%) combined with moderate temperatures (60-77°F / 16-25°C)."
        ],
        "immediate_actions": [
            "Assess crop growth stage: rust appearing after grain fill rarely impacts economic yield.",
            "Ensure adequate soil potassium to help plants tolerate rust stress."
        ],
        "preventative_measures": [
            "Plant resistant corn hybrids with specific Rp genes.",
            "Rotate crops annually with non-host species (legumes, brassicas).",
            "Plant early to allow the crop to mature before airborne spore flights arrive."
        ],
        "organic_controls": [
            "Preventative neem or biological Bacillus amyloliquefaciens foliar treatments on sweet corn."
        ],
        "expert_consultation": "Contact your crop consultant if rust covers >10% of ear leaf area before tasseling stage."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "plant": "Corn (Maize)",
        "disease": "Northern Corn Leaf Blight (Exserohilum turcicum)",
        "is_healthy": False,
        "severity": "High",
        "description": "A serious fungal blight forming large cigar-shaped lesions that severely reduce photosynthetic capacity and grain fill.",
        "symptoms": [
            "Long, elliptical, grayish-green or tan lesions (1 to 6 inches long) with blunt ends (cigar-shaped).",
            "Dark spore zones forming in concentric bands within lesions during humid weather.",
            "Extensive leaf blighting starting from lower canopy and moving upward."
        ],
        "causes": [
            "Overwintering mycelia in unburied corn residue.",
            "Frequent rainfall, heavy morning dew, and moderate temperatures (65-80°F / 18-27°C)."
        ],
        "immediate_actions": [
            "Avoid excessive overhead irrigation that prolongs canopy wetness.",
            "Document lesion progression relative to the ear leaf."
        ],
        "preventative_measures": [
            "Select resistant hybrids with Ht gene resistance.",
            "Practice minimum 1-year crop rotation out of corn.",
            "Incorporate crop residue post-harvest through conservation tillage to accelerate decomposition."
        ],
        "organic_controls": [
            "Bio-fungicides applied at early growth stages if weather forecast favors prolonged humidity."
        ],
        "expert_consultation": "Seek extension agronomist guidance if lesions appear on or above the ear leaf before silking."
    },
    "Corn_(maize)___healthy": {
        "plant": "Corn (Maize)",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "description": "Corn canopy exhibits strong structural integrity, uniform chlorophyll distribution, and no pathogen sporulation.",
        "symptoms": [
            "Vigorous green leaves with distinct parallel venation.",
            "Sturdy leaf sheaths and absence of foliar lesions or gall formation."
        ],
        "causes": [
            "Balanced nitrogen-phosphorus-potassium fertility and proper soil drainage."
        ],
        "immediate_actions": [
            "Continue standard agronomic scouting and crop monitoring."
        ],
        "preventative_measures": [
            "Side-dress nitrogen according to soil test recommendations during V6 growth stage.",
            "Ensure weed management to eliminate alternative hosts for insect vectors."
        ],
        "organic_controls": [
            "Beneficial cover cropping (crimson clover or hairy vetch) during off-seasons."
        ],
        "expert_consultation": "Standard agronomic tissue sampling during tasseling to verify micronutrient levels."
    },
    "Grape___Black_rot": {
        "plant": "Grape",
        "disease": "Black Rot (Guignardia bidwellii)",
        "is_healthy": False,
        "severity": "High",
        "description": "A widespread fungal disease devastating vineyards, turning succulent berries into shriveled black mummies and spotting foliage.",
        "symptoms": [
            "Small circular, tan leaf spots bordered by dark brown margins with tiny black fruiting bodies (pycnidia).",
            "Infected fruit shrivels, turns bluish-black, and dries into a hard wrinkled mummy attached to the cluster.",
            "Black elliptical cankers on shoots and tendrils."
        ],
        "causes": [
            "Overwintering mummies left hanging in trellises or dropped on vineyard floor.",
            "Warm, wet weather (80°F / 27°C with continuous leaf wetness for 6+ hours)."
        ],
        "immediate_actions": [
            "Carefully hand-remove and destroy all mummified grape clusters and infected canes.",
            "Canopy management: tuck shoots and pull basal leaves around fruit clusters to speed air drying."
        ],
        "preventative_measures": [
            "Maintain immaculate vineyard floor sanitation.",
            "Train vines on high trellis systems to improve sun exposure and airflow.",
            "Apply protective fungicides (mancozeb, captan, or copper) starting at 1-3 inch shoot growth."
        ],
        "organic_controls": [
            "Lime sulfur applications during delayed dormant stage followed by copper soap during early shoot growth."
        ],
        "expert_consultation": "Contact your viticulture extension specialist if black rot is detected in >5% of fruit clusters before veraison."
    },
    "Grape___healthy": {
        "plant": "Grape",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "description": "The grape foliage demonstrates healthy palmate leaf structure, deep green hue, and flawless cuticle layer.",
        "symptoms": [
            "Bright, uniform green palmate leaves with healthy serrated margins.",
            "Vigorous shoot growth and clean tendril formation."
        ],
        "causes": [
            "Effective canopy aeration, adequate sunlight, and proactive pest monitoring."
        ],
        "immediate_actions": [
            "Continue canopy maintenance, shoot positioning, and routine vine scouting."
        ],
        "preventative_measures": [
            "Perform summer shoot thinning to maintain a 1-2 shoot per cluster density.",
            "Avoid high nitrogen applications that promote excessive vegetative shading."
        ],
        "organic_controls": [
            "Foliar seaweed extract applications to enhance vine natural defense mechanisms."
        ],
        "expert_consultation": "Routine petiole nutrient analysis during full bloom."
    },
    "Pepper__bell___Bacterial_spot": {
        "plant": "Pepper (Bell)",
        "disease": "Bacterial Spot (Xanthomonas campestris pv. vesicatoria)",
        "is_healthy": False,
        "severity": "High",
        "description": "A destructive bacterial pathogen affecting peppers and tomatoes, causing water-soaked spots that blister and lead to severe defoliation and sunscalded fruit.",
        "symptoms": [
            "Small water-soaked, circular to irregular lesions on leaves that turn dark brown with yellow halos.",
            "Leaves turn yellow and drop prematurely, leaving fruit exposed to sunscald.",
            "Raised, blister-like or scabby warts on the fruit surface."
        ],
        "causes": [
            "Infected seed or infected transplant stock.",
            "Splashing water, overhead sprinklers, and handling wet plants.",
            "High temperatures (75-90°F / 24-32°C) combined with high relative humidity."
        ],
        "immediate_actions": [
            "Immediately stop overhead watering; use drip or soaker hose irrigation exclusively.",
            "Do not prune, harvest, or walk through pepper rows when foliage is wet.",
            "Rogue out and destroy severely diseased plants to prevent bacterial spread."
        ],
        "preventative_measures": [
            "Only use certified disease-free, hot-water treated seeds and clean transplants.",
            "Practice a minimum 2-to-3-year crop rotation out of solanaceous crops (tomatoes, peppers, potatoes, eggplants).",
            "Apply fixed copper sprays combined with mancozeb or bio-bactericides early in the vegetative cycle."
        ],
        "organic_controls": [
            "Copper hydroxide formulations tank-mixed with Bacillus subtilis sprays."
        ],
        "expert_consultation": "Consult agricultural extension if commercial pepper fields experience widespread spot development during early flowering."
    },
    "Pepper__bell___healthy": {
        "plant": "Pepper (Bell)",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "description": "The bell pepper foliage displays glossy dark green coloration, firm turgor pressure, and no signs of bacterial or viral stress.",
        "symptoms": [
            "Smooth, glossy green leaves with intact leaf tips.",
            "Uniform growth without mottling, blistering, or defoliation."
        ],
        "causes": [
            "Adequate calcium in soil, even moisture distribution, and proper temperature regulation."
        ],
        "immediate_actions": [
            "Maintain consistent soil moisture to prevent blossom end rot later in the season."
        ],
        "preventative_measures": [
            "Mulch with clean straw or black plastic mulch to suppress weeds and retain soil moisture.",
            "Maintain soil pH between 6.2 and 6.8 with adequate calcium."
        ],
        "organic_controls": [
            "Weekly insect scouting for aphids, thrips, and mites that transmit plant viruses."
        ],
        "expert_consultation": "Consult local master gardeners if unexplained leaf drop occurs during temperature extremes (>90°F / 32°C)."
    },
    "Potato___Early_blight": {
        "plant": "Potato",
        "disease": "Early Blight (Alternaria solani)",
        "is_healthy": False,
        "severity": "Moderate to High",
        "description": "A common fungal disease causing concentric target-board lesions on older potato foliage, reducing photosynthetic capacity and tuber bulking.",
        "symptoms": [
            "Small dark brown to black spots with distinctive concentric rings ('target-board' pattern).",
            "Yellow halo (chlorosis) surrounding the dark target lesions.",
            "Lower and older leaves are infected first, eventually withering and hanging dead on the stem."
        ],
        "causes": [
            "Alternaria spores surviving in infected soil debris and solanaceous weeds (nightshade).",
            "Alternating wet and dry periods with warm temperatures (75-85°F / 24-29°C).",
            "Stressed plants due to nitrogen deficiency, nematode injury, or heavy tuber load."
        ],
        "immediate_actions": [
            "Prune off infected lower leaves using clean shears to slow upward progression.",
            "Apply drip irrigation to keep upper canopy dry.",
            "Provide light balanced fertilization to relieve nutritional stress."
        ],
        "preventative_measures": [
            "Rotate potato fields with non-solanaceous crops (grasses, brassicas) for 3 years.",
            "Ensure certified disease-free seed tubers are used.",
            "Apply protective fungicides (chlorothalonil, copper, or azoxystrobin) prior to canopy closure."
        ],
        "organic_controls": [
            "Copper octanoate (copper soap) or bio-fungicides applied at first detection."
        ],
        "expert_consultation": "Contact your crop extension office if symptoms spread rapidly to upper leaves before tuber bulking is complete."
    },
    "Potato___Late_blight": {
        "plant": "Potato",
        "disease": "Late Blight (Phytophthora infestans)",
        "is_healthy": False,
        "severity": "Very High (Emergency)",
        "description": "An aggressive, historic water-mold (oomycete) that caused the Irish Potato Famine. It can destroy entire fields within a week under cool, moist conditions.",
        "symptoms": [
            "Irregular, water-soaked dark green to purplish-black lesions that expand rapidly.",
            "Delicate white, cottony fungal-like growth visible on the underside of leaves in moist morning conditions.",
            "Foul decaying odor and rapidly collapsing stems and vines.",
            "Tubers develop sunken, dry, reddish-brown granular rot."
        ],
        "causes": [
            "Cool, wet weather (60-70°F / 15-21°C) with persistent cloud cover and humidity >90%.",
            "Infected seed tubers or volunteer potato culls from previous seasons."
        ],
        "immediate_actions": [
            "URGENT: Immediately notify your local agricultural extension service.",
            "Destroy severely infected plants immediately (bag and solarize; do not compost).",
            "Do not harvest tubers from infected plants until vines have been killed and dried for 2+ weeks."
        ],
        "preventative_measures": [
            "Destroy all potato cull piles and volunteer plants in spring.",
            "Plant late-blight resistant potato varieties (e.g., Defender, Elba, Sarpo Mira).",
            "Adhere to commercial blight-warning forecasting models (such as Blitecast)."
        ],
        "organic_controls": [
            "Fixed copper sprays applied strictly as a preventative shield prior to spore arrival."
        ],
        "expert_consultation": "IMMEDIATELY consult your regional agricultural extension agent; late blight is an internationally monitored community pathogen."
    },
    "Potato___healthy": {
        "plant": "Potato",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "description": "Potato foliage shows vigorous compound leaves, robust stems, and no signs of bacterial wilt, virus mosaic, or blight.",
        "symptoms": [
            "Deep green compound foliage with intact margins.",
            "Sturdy upright stems supporting uniform flower and leaf development."
        ],
        "causes": [
            "Loose, well-draining acidic soil (pH 5.2-6.0), adequate hilling, and proper nutrient supply."
        ],
        "immediate_actions": [
            "Continue standard hilling practices to protect emerging tubers from sun exposure."
        ],
        "preventative_measures": [
            "Maintain even moisture (about 1-2 inches per week) especially during flowering and tuberization.",
            "Hill soil around plant stems when plants reach 8-10 inches tall."
        ],
        "organic_controls": [
            "Inspect weekly under leaves for Colorado Potato Beetle eggs and hand-pick adults."
        ],
        "expert_consultation": "Soil fertility test every 2 years to monitor phosphorus and potassium availability."
    },
    "Tomato___Early_blight": {
        "plant": "Tomato",
        "disease": "Early Blight (Alternaria linariae / solani)",
        "is_healthy": False,
        "severity": "Moderate to High",
        "description": "A very common fungal disease affecting tomatoes in home gardens and commercial fields, causing concentric rings on leaves and stem collars.",
        "symptoms": [
            "Dark brown to black spots with distinctive concentric rings ('target' pattern).",
            "Leaf tissue surrounding spots turns yellow; leaves gradually dry and drop from the bottom up.",
            "Dark, sunken, leathery cankers near the base of the stem (collar rot on seedlings)."
        ],
        "causes": [
            "Fungus overwinters on contaminated crop residue and stakes in the soil.",
            "Rain splash bouncing soil spores onto lower foliage.",
            "Warm temperatures (75-85°F / 24-29°C) and heavy dew or overhead irrigation."
        ],
        "immediate_actions": [
            "Prune away all infected leaves near the bottom of the plant; sanitize pruners with alcohol.",
            "Apply a 2-3 inch organic mulch (straw, untreated wood chips) beneath plants to block soil splashing.",
            "Water exclusively at soil level using a drip line or watering wand."
        ],
        "preventative_measures": [
            "Space tomato plants 24-36 inches apart to allow strong air circulation.",
            "Stake or cage tomatoes to keep foliage off wet soil.",
            "Rotate crops annually so tomatoes or related nightshades are not planted in the same spot for 3 years."
        ],
        "organic_controls": [
            "Liquid copper octanoate or Bacillus amyloliquefaciens biological foliar sprays applied every 7-10 days."
        ],
        "expert_consultation": "Consult extension experts if over 40% of the plant canopy is infected before the first fruit cluster matures."
    },
    "Tomato___Late_blight": {
        "plant": "Tomato",
        "disease": "Late Blight (Phytophthora infestans)",
        "is_healthy": False,
        "severity": "Very High (Emergency)",
        "description": "An aggressive, fast-moving oomycete that turns tomato foliage and fruit water-soaked and brown within days, often devastating gardens during cool, damp weather.",
        "symptoms": [
            "Large, irregular water-soaked pale to greasy-dark lesions on leaves and stems.",
            "White fuzzy or downy mildew growth appearing on the underside of infected leaves in high humidity.",
            "Tomatoes develop large, greasy, firm, olive-brown patches.",
            "Rapid plant collapse resembling severe frost damage."
        ],
        "causes": [
            "Airborne sporangia carried for miles on moist wind currents.",
            "Cool temperatures (60-70°F / 15-21°C) with persistent dampness, fog, or prolonged rainfall."
        ],
        "immediate_actions": [
            "Act quickly: remove and bag infected plants during dry weather; do NOT compost them.",
            "Disinfect all cages, stakes, and tools with 10% bleach or rubbing alcohol.",
            "Alert neighboring gardeners and local extension agents."
        ],
        "preventative_measures": [
            "Plant resistant tomato cultivars (e.g., Mountain Magic, Defiant, Iron Lady, Plum Regal).",
            "Avoid planting near potatoes or overwintered compost piles.",
            "Monitor regional blight trackers (e.g., USAblight)."
        ],
        "organic_controls": [
            "Copper fungicides applied strictly before symptoms appear when regional blight warnings are issued."
        ],
        "expert_consultation": "Report sightings immediately to your local county extension office to help protect regional tomato crops."
    },
    "Tomato___Leaf_Mold": {
        "plant": "Tomato",
        "disease": "Leaf Mold (Passalora fulva)",
        "is_healthy": False,
        "severity": "Moderate",
        "description": "A fungal foliar disease common in high tunnels and greenhouses with elevated humidity, creating pale yellow leaf patches and velvety olive spores.",
        "symptoms": [
            "Pale greenish-yellow chlorotic spots with indistinct margins on upper leaf surfaces.",
            "Olive-green to grayish-brown velvety mold growth on the corresponding lower leaf surface.",
            "Infected leaves wither, turn yellow-brown, and drop prematurely."
        ],
        "causes": [
            "High relative humidity (>85%) combined with temperatures between 70-75°F (21-24°C).",
            "Inadequate ventilation inside protected structures, hoop houses, or dense outdoor plantings."
        ],
        "immediate_actions": [
            "Increase ventilation: open greenhouse sides, run circulating fans, or prune excess suckers.",
            "Avoid wetting foliage during irrigation."
        ],
        "preventative_measures": [
            "Ensure spacing of at least 30 inches between plants in greenhouses.",
            "Choose resistant varieties containing Cf genes.",
            "Heat and ventilate greenhouses in late afternoon to drive off moist air."
        ],
        "organic_controls": [
            "Bio-fungicides based on Trichoderma harzianum or potassium bicarbonate sprays."
        ],
        "expert_consultation": "Contact greenhouse extension specialists if humidity control cannot arrest disease progression."
    },
    "Tomato___Septoria_leaf_spot": {
        "plant": "Tomato",
        "disease": "Septoria Leaf Spot (Septoria lycopersici)",
        "is_healthy": False,
        "severity": "Moderate to High",
        "description": "One of the most destructive foliar diseases of tomato, producing countless tiny circular spots that cause progressive bottom-up defoliation.",
        "symptoms": [
            "Numerous small, circular spots (1/16 to 1/8 inch) with dark brown margins and sunken white or gray centers.",
            "Tiny black specks (pycnidia) visible inside the center of mature spots under magnification.",
            "Progressive yellowing and browning of leaves starting from the lowest branches."
        ],
        "causes": [
            "Overwintering in solanaceous weeds (horsenettle) and tomato crop residue.",
            "Prolonged leaf wetness and warm temperatures (68-77°F / 20-25°C).",
            "Rain splash and overhead sprinklers."
        ],
        "immediate_actions": [
            "Strip off infected lower foliage immediately and discard in trash.",
            "Add heavy mulch layer to stop spore splash from soil onto low-hanging leaves.",
            "Ensure plants are staked upright."
        ],
        "preventative_measures": [
            "3-year crop rotation out of tomatoes, potatoes, eggplants, and peppers.",
            "Prune lowest 12 inches of suckers and foliage once plants are established to eliminate soil contact.",
            "Apply copper-based or chlorothalonil fungicides at the first sign of lower spotting."
        ],
        "organic_controls": [
            "Copper soap (copper octanoate) combined with biological foliar sprays applied every 7-10 days."
        ],
        "expert_consultation": "Consult extension services if defoliation reaches fruit clusters early in harvest."
    },
    "Tomato___Yellow_Leaf_Curl_Virus": {
        "plant": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus (TYLCV)",
        "is_healthy": False,
        "severity": "High to Severe",
        "description": "A devastating viral disease transmitted by the silverleaf whitefly (Bemisia tabaci), causing stunted growth, cupped leaves, and total fruit loss if infected early.",
        "symptoms": [
            "Severe upward curling and cupping of leaflet margins.",
            "Pronounced yellowing (chlorosis) along leaf margins and interveinal areas.",
            "Marked stunting of the plant with short internodes, giving a bushy appearance.",
            "Flower abortion resulting in little or no fruit set."
        ],
        "causes": [
            "Persistent transmission by silverleaf whiteflies feeding on plant sap.",
            "Presence of alternative weed hosts (nightshades, mallows, datura).",
            "Warm, arid or tropical microclimates favoring rapid whitefly multiplication."
        ],
        "immediate_actions": [
            "Uproot and bag infected plants immediately in plastic sacks to prevent whiteflies from escaping to healthy plants.",
            "Inspect underside of leaves on nearby plants for tiny moth-like whiteflies.",
            "Install yellow sticky traps to detect and trap adult whiteflies."
        ],
        "preventative_measures": [
            "Plant TYLCV-resistant tomato cultivars (e.g., Camaro, Charger, Defiant, Skyway).",
            "Use fine mesh insect netting (50 mesh) over nursery beds and high tunnels.",
            "Eradicate broadleaf weeds surrounding garden borders."
        ],
        "organic_controls": [
            "Insecticidal soap or horticultural mineral oil sprays targeted at leaf undersides to smother whitefly nymphs.",
            "Introduce beneficial predatory insects like lacewings or Encarsia formosa parasitic wasps."
        ],
        "expert_consultation": "Notify your agricultural extension office or state department of agriculture if confirmed in regions where TYLCV is regulated or emerging."
    },
    "Tomato___healthy": {
        "plant": "Tomato",
        "disease": "Healthy Leaf",
        "is_healthy": True,
        "severity": "None",
        "description": "Tomato foliage shows robust pinnate foliage, deep green coloration, healthy glandular hairs, and zero signs of pathogen damage.",
        "symptoms": [
            "Vibrant green leaves with crisp edges and characteristic healthy aroma.",
            "Sturdy central stems with active apical meristem and flower development."
        ],
        "causes": [
            "Ideal balance of sunshine (6-8 hours), well-aerated fertile soil, and disciplined watering."
        ],
        "immediate_actions": [
            "Keep up excellent plant maintenance: consistent watering and light sucker pruning."
        ],
        "preventative_measures": [
            "Water at the base of the plant early in the morning.",
            "Maintain 2-3 inches of straw mulch to prevent soilborne pathogens from splashing onto lower leaves.",
            "Support plants with sturdy stakes, trellises, or heavy tomato cages."
        ],
        "organic_controls": [
            "Proactive foliar compost tea application every 2 weeks to foster beneficial protective phyllosphere microbes."
        ],
        "expert_consultation": "Periodic soil testing every 2 seasons to optimize calcium:magnesium ratio for blossom-end rot prevention."
    }
}

# Fallback recommendation for out-of-distribution or uncertain predictions
UNKNOWN_RECOMMENDATION: Dict[str, Any] = {
    "plant": "Unknown / Unsupported Plant",
    "disease": "Inconclusive / Low Confidence Prediction",
    "is_healthy": False,
    "severity": "Uncertain",
    "description": "The AI model could not identify the plant or disease with sufficient confidence (>60%). This image may be out-of-distribution, blurry, poorly lit, or represent a plant species not present in the model's training dataset.",
    "symptoms": [
        "Visual features do not strongly match any supported agricultural disease classes.",
        "Possible factors: blurry camera focus, glare, extreme close-up, or non-plant subject."
    ],
    "causes": [
        "Out-of-distribution image or rare physiological condition not covered in the training set.",
        "Multiple overlapping symptoms (e.g., nutrient deficiency combined with insect feeding)."
    ],
    "immediate_actions": [
        "Take a fresh photograph in clear natural daylight, ensuring the leaf is flat, in sharp focus, and centered.",
        "Check both the upper and lower leaf surfaces for signs of webbing, insects, or fungal spores.",
        "Do NOT apply chemical treatments without a verified diagnosis."
    ],
    "preventative_measures": [
        "Maintain standard sanitary gardening hygiene (clean tools, drip irrigation).",
        "Monitor the affected plant over the next 48-72 hours to see if symptoms progress."
    ],
    "organic_controls": [
        "Isolate suspicious potted plants from others until symptoms are understood."
    ],
    "expert_consultation": "Bring a physical leaf sample sealed in a clear plastic bag to your local Agricultural Extension Office or a university plant pathology diagnostic clinic for expert evaluation."
}


def parse_class_name(raw_class: str) -> tuple[str, str]:
    """
    Parses a raw dataset class name like 'Tomato___Early_blight' into
    human-readable (Plant Name, Disease Name).
    """
    if "___" in raw_class:
        parts = raw_class.split("___", 1)
        plant = parts[0].replace("_", " ").strip()
        disease = parts[1].replace("_", " ").strip()
        # Clean up specific formatting
        if plant.lower() == "pepper bell":
            plant = "Pepper (Bell)"
        elif plant.lower() == "corn (maize)":
            plant = "Corn (Maize)"
        if disease.lower() == "healthy":
            disease = "Healthy Leaf"
        return plant, disease
    return "Plant", raw_class.replace("_", " ").strip()


def get_recommendation(raw_class: str, confidence: float = 1.0) -> Dict[str, Any]:
    """
    Retrieves the complete recommendation profile for a given class label.
    If confidence is below threshold (< 0.60) or class is unknown,
    returns an informative cautionary profile with the raw prediction noted.
    """
    if confidence < 0.60 or raw_class not in DISEASE_KNOWLEDGE_BASE:
        rec = UNKNOWN_RECOMMENDATION.copy()
        if raw_class in DISEASE_KNOWLEDGE_BASE:
            known = DISEASE_KNOWLEDGE_BASE[raw_class]
            rec["candidate_prediction"] = f"{known['plant']} - {known['disease']}"
            rec["confidence_warning"] = (
                f"Model candidate is {known['plant']} ({known['disease']}) "
                f"with low confidence ({confidence * 100:.1f}%). Verification is strongly advised."
            )
        else:
            rec["candidate_prediction"] = raw_class
            rec["confidence_warning"] = "Class is not recognized in the knowledge base."
        return rec

    return DISEASE_KNOWLEDGE_BASE[raw_class]


def get_all_library_entries() -> List[Dict[str, Any]]:
    """
    Returns all supported disease and healthy entries formatted
    for the frontend Disease Library.
    """
    entries = []
    for class_key, profile in DISEASE_KNOWLEDGE_BASE.items():
        entry = profile.copy()
        entry["class_key"] = class_key
        entries.append(entry)
    # Sort alphabetically by plant, then disease
    entries.sort(key=lambda x: (x["plant"], x["disease"]))
    return entries
