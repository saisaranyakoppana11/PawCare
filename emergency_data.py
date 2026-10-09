"""
PawCare - Emergency Guidance Data
Provides general situational first-response safety guidance only.
Does NOT replace licensed veterinary examination, diagnosis, or care.
"""

EMERGENCY_DISCLAIMER = (
    "IMPORTANT DISCLAIMER: This guide provides general situational first-response "
    "safety guidance only and does NOT constitute veterinary diagnosis or medical treatment. "
    "In any emergency, always contact or transport your pet to a licensed veterinarian "
    "or 24-hour emergency animal hospital immediately."
)

EMERGENCY_CATEGORIES = [
    {
        "id": "breathing-difficulty",
        "title": "Breathing Difficulty & Respiratory Distress",
        "icon": "🫁",
        "urgency": "Critical Emergency",
        "badge_class": "bg-danger",
        "description": (
            "Signs include rapid or shallow breathing, loud wheezing, flared nostrils, "
            "neck stretched out with open-mouth breathing (especially in cats), pale, blue, "
            "or purple gums, or visible struggle to draw breath."
        ),
        "immediate_actions": [
            "Keep the pet and yourself as calm and quiet as possible; stress increases oxygen demand.",
            "Loosen or remove collars, harnesses, or sweaters immediately.",
            "Ensure the pet is in a cool, well-ventilated space with fresh airflow.",
            "Keep the pet's neck extended gently to open airways, avoiding pressure on the throat or chest.",
            "Call the nearest emergency veterinary clinic right away so they can prepare oxygen before your arrival."
        ],
        "things_to_avoid": [
            "DO NOT attempt mouth-to-snout resuscitation unless pet is completely unresponsive and you are trained.",
            "DO NOT force the pet into an airtight, enclosed box or dark container.",
            "DO NOT force the pet to drink water or eat.",
            "DO NOT compress the ribcage or apply tight wraps around the torso."
        ],
        "vet_recommendation": (
            "Respiratory distress is a life-threatening emergency. Transport your pet to the nearest "
            "veterinary facility immediately with air conditioning running in the car."
        )
    },
    {
        "id": "bleeding",
        "title": "Severe Bleeding & Open Wounds",
        "icon": "🩸",
        "urgency": "Critical Emergency",
        "badge_class": "bg-danger",
        "description": (
            "Arterial or severe venous bleeding resulting from deep cuts, bites, vehicle accidents, "
            "or puncture wounds where blood is spurting, pooling rapidly, or soaking through bandages."
        ),
        "immediate_actions": [
            "Apply firm, continuous direct pressure over the wound using a clean towel, gauze pad, or cloth.",
            "Maintain steady pressure for at least 3 to 5 minutes without lifting to check if it has stopped.",
            "If blood soaks through, add another layer of cloth on top without removing the original compress.",
            "Keep the pet warm with a blanket to prevent shock.",
            "If an extremity is bleeding, gently elevate the limb slightly while maintaining direct pressure."
        ],
        "things_to_avoid": [
            "DO NOT apply a tight tourniquet unless instructed by a vet, as improper use causes tissue necrosis.",
            "DO NOT probe or clean deep, actively bleeding wounds with antiseptics or alcohol.",
            "DO NOT remove deeply embedded objects (sticks, glass) yourself, as they may be plugging blood vessels.",
            "DO NOT allow the pet to lick, bite, or scratch at the wound."
        ],
        "vet_recommendation": (
            "Uncontrolled bleeding leads quickly to hypovolemic shock. Maintain continuous direct pressure "
            "while transporting the pet directly to emergency veterinary care."
        )
    },
    {
        "id": "poisoning",
        "title": "Poisoning & Toxic Ingestion Concern",
        "icon": "⚠️",
        "urgency": "High Urgency",
        "badge_class": "bg-danger",
        "description": (
            "Suspected or witnessed ingestion or contact with known pet toxins (e.g. human medications, "
            "chocolate, xylitol/sweetener, rodenticides, lilies for cats, onions/garlic, grapes/raisins, "
            "antifreeze, household cleaning chemicals)."
        ),
        "immediate_actions": [
            "Remove the pet from any remaining toxic substance immediately.",
            "Safely collect the packaging, bottle, plant sample, or remaining material to show the veterinarian.",
            "Note the exact time of ingestion and the estimated quantity ingested.",
            "If a toxin is on the skin or fur, prevent licking and rinse with lukewarm water.",
            "Contact your veterinarian or an animal poison hotline immediately for substance-specific guidance."
        ],
        "things_to_avoid": [
            "DO NOT induce vomiting with hydrogen peroxide or salt unless explicitly instructed by a veterinarian (can cause fatal lung aspiration or stomach rupture).",
            "DO NOT give milk, raw eggs, oils, or home remedies without professional advice.",
            "DO NOT wait for symptoms to appear before seeking professional help; many toxins cause irreversible damage within hours."
        ],
        "vet_recommendation": (
            "Treatment windows for toxins like xylitol, lilies, and antifreeze are extremely narrow. Contact your vet or emergency clinic immediately."
        )
    },
    {
        "id": "heat-stroke",
        "title": "Heat-Related Emergency & Heatstroke",
        "icon": "☀️",
        "urgency": "Critical Emergency",
        "badge_class": "bg-danger",
        "description": (
            "Elevated core temperature resulting from hot weather, being left in a parked vehicle, or "
            "strenuous exercise. Symptoms include excessive drooling, deep panting, bright red or dark gums, "
            "unsteadiness, vomiting, or collapse."
        ),
        "immediate_actions": [
            "Move the pet immediately into a shaded, air-conditioned indoor area or car with active ventilation.",
            "Apply cool (NOT ice-cold) tap water to the pet's paws, abdomen, groin, and ears using soaked towels.",
            "Turn on a fan directed toward the pet to assist evaporative cooling.",
            "Offer small amounts of cool, fresh water to lap voluntarily if fully conscious.",
            "Stop active cooling once the pet's breathing calms or body temperature nears 39.5°C (103°F) to prevent hypothermia."
        ],
        "things_to_avoid": [
            "DO NOT use ice or ice-cold water bath; extreme cold causes peripheral blood vessels to constrict, trapping heat inside internal organs.",
            "DO NOT force water down the throat of an unconscious or disoriented pet.",
            "DO NOT leave damp towels resting continuously on the pet without refreshing, as they can trap heat."
        ],
        "vet_recommendation": (
            "Heatstroke causes delayed organ failure and internal bleeding hours later. Even if your pet appears "
            "recovered, an emergency veterinary evaluation is essential."
        )
    },
    {
        "id": "seizures",
        "title": "Seizure-Like Event & Tremors",
        "icon": "⚡",
        "urgency": "High Urgency",
        "badge_class": "bg-danger",
        "description": (
            "Sudden involuntary muscle contractions, paddling of legs, jaw chomping, drooling, "
            "loss of consciousness, involuntary urination/defecation, or disorientation lasting from seconds to minutes."
        ),
        "immediate_actions": [
            "Clear away hard, sharp, or hazardous furniture to prevent secondary physical injury.",
            "Place soft cushions, towels, or blankets on the floor around the pet.",
            "Dim bright lights, turn off TVs/loud audio, and keep the environment calm and quiet.",
            "Time the seizure accurately with a phone stopwatch and take a brief video if possible for the vet.",
            "Speak in a gentle, reassuring tone as the pet recovers from the post-ictal disorientation."
        ],
        "things_to_avoid": [
            "DO NOT place your hands, spoons, or fingers anywhere near the pet's mouth (animals do not swallow their tongues, but can bite severely by reflex).",
            "DO NOT attempt to hold down or forcefully restrain the convulsing pet.",
            "DO NOT offer food, treats, or water immediately following a seizure until full consciousness returns."
        ],
        "vet_recommendation": (
            "If a seizure lasts longer than 3 minutes, or if your pet has multiple seizures in a 24-hour period "
            "(cluster seizures), this is a critical emergency requiring immediate IV anticonvulsant treatment."
        )
    },
    {
        "id": "injury",
        "title": "Physical Trauma, Falls & Suspected Fractures",
        "icon": "🩹",
        "urgency": "High Urgency",
        "badge_class": "bg-warning text-dark",
        "description": (
            "Trauma resulting from vehicular impact, falls from balconies/stairs, or severe fights. "
            "Signs include inability to bear weight on limbs, abnormal bone angles, swelling, whimpering, or immobility."
        ),
        "immediate_actions": [
            "Approach slowly and cautiously; injured, frightened animals may bite defensively out of intense pain.",
            "Keep the pet's spine, neck, and body as flat and straight as possible during handling.",
            "Use a rigid support such as a firm board, stretcher, or thick folded blanket to lift the pet.",
            "Cover open fractures with a clean, dry, sterile cloth to keep out debris.",
            "Keep the pet warm with a light blanket during transit."
        ],
        "things_to_avoid": [
            "DO NOT attempt to push back exposed bones or set splints yourself.",
            "DO NOT manipulate painful limbs unnecessarily.",
            "DO NOT give human pain medications (e.g. Paracetamol, Ibuprofen, Aspirin) — these are extremely toxic or fatal to pets.",
            "DO NOT allow the pet to walk on injured limbs."
        ],
        "vet_recommendation": (
            "Trauma often involves hidden internal injuries (such as ruptured bladders, diaphragmatic hernias, or pneumothorax). Prompt veterinary imaging is crucial."
        )
    },
    {
        "id": "choking",
        "title": "Choking Concern & Airway Obstruction",
        "icon": "🛑",
        "urgency": "Critical Emergency",
        "badge_class": "bg-danger",
        "description": (
            "Pet is pawing violently at the mouth, gagging, coughing violently, salivating heavily, "
            "wheezing with pale or bluish gums, or unable to draw breath after chewing on a toy, bone, or small object."
        ),
        "immediate_actions": [
            "Carefully look inside the mouth ONLY if the pet is calm enough to not bite.",
            "If a foreign object is clearly visible and loose in the front of the mouth, carefully sweep it out with a finger or blunt tweezers.",
            "For small pets: gently hold upside down by the thighs for a few seconds to let gravity assist.",
            "For larger dogs: perform a modified Heimlich maneuver by placing hands under the abdomen behind the ribcage and thrusting firmly inward and upward 3-5 times.",
            "Rush to emergency veterinary care immediately."
        ],
        "things_to_avoid": [
            "DO NOT blindly shove fingers down the throat; this can push the object deeper and wedge it tighter.",
            "DO NOT pull on string, needles, or threads caught in the throat (can lacerate the esophagus or intestines).",
            "DO NOT hesitate or delay emergency transport if the object cannot be removed immediately in seconds."
        ],
        "vet_recommendation": (
            "Even if the object is dislodged, throat swelling or pulmonary edema can develop. Have a veterinarian inspect the airway."
        )
    },
    {
        "id": "vomiting-diarrhea",
        "title": "Severe Vomiting & Dehydration",
        "icon": "💧",
        "urgency": "High Urgency",
        "badge_class": "bg-warning text-dark",
        "description": (
            "Repeated vomiting episodes (more than 3 times in a few hours), projectile vomiting, "
            "presence of blood (bright red or dark coffee-ground appearance), dry tacky gums, or extreme lethargy."
        ),
        "immediate_actions": [
            "Temporarily withhold solid food for a few hours to allow the gastrointestinal tract to rest.",
            "Offer small, controlled sips of water (a few teaspoons) rather than allowing large gulps that trigger vomiting.",
            "Monitor gum moisture: press gently on the gums; if tacky, pale, or color returns slowly (>2s), dehydration is present.",
            "Note the frequency, volume, color, and presence of foreign materials or blood.",
            "Keep the pet in a warm, quiet, clean resting area."
        ],
        "things_to_avoid": [
            "DO NOT administer human anti-diarrhea or anti-nausea medications without veterinary prescription.",
            "DO NOT force food or milk on a nauseated pet.",
            "DO NOT delay veterinary care if your pet is a puppy or kitten, as young animals dehydrate rapidly within hours."
        ],
        "vet_recommendation": (
            "Severe persistent vomiting can indicate foreign body bowel obstruction, pancreatitis, or parvovirus. Seek prompt veterinary attention."
        )
    },
    {
        "id": "unconsciousness",
        "title": "Unconsciousness & Sudden Collapse",
        "icon": "🌀",
        "urgency": "Critical Emergency",
        "badge_class": "bg-danger",
        "description": (
            "Sudden loss of consciousness, unresponsiveness to verbal calls or physical touch, "
            "limp muscle tone, or sudden collapse while walking or exercising."
        ),
        "immediate_actions": [
            "Check for a heartbeat or femoral pulse (located on the inside of the upper rear thigh).",
            "Check for breathing by watching chest movement or feeling for airflow near the nostrils.",
            "Ensure the head and neck are straight and airways are clear of vomit, mucus, or foreign objects.",
            "Keep the pet level or with hindquarters slightly elevated to promote cerebral blood flow.",
            "Wrap gently in a warm blanket to conserve body heat during transport."
        ],
        "things_to_avoid": [
            "DO NOT shake, slap, or violently jostle the unconscious pet.",
            "DO NOT pour liquids or put medications into the mouth of an unconscious pet.",
            "DO NOT attempt chest compressions if a pulse is detectable."
        ],
        "vet_recommendation": (
            "Sudden collapse can indicate cardiac arrest, internal hemorrhage, severe hypoglycemia, or stroke. Immediate emergency veterinary intervention is vital."
        )
    },
    {
        "id": "other-urgent",
        "title": "Other Urgent & Acute Situations",
        "icon": "🚨",
        "urgency": "Urgent Care",
        "badge_class": "bg-secondary",
        "description": (
            "Includes severe eye injuries/corneal scratches, painful swollen abdomen, inability to urinate "
            "(especially male cats — a critical emergency), sudden paralysis of hind legs, or severe allergic swelling."
        ),
        "immediate_actions": [
            "Inability to urinate in male cats: Do NOT press on the bladder; transport to vet immediately.",
            "Eye injury: Prevent the pet from rubbing or scratching the eye; use an Elizabethan collar if available.",
            "Bloated abdomen with retching in dogs (suspected GDV / Gastric Torsion): Treat as a critical surgical emergency.",
            "Keep the pet calm, secure, and comfortably supported.",
            "Call the emergency clinic to notify them of your arrival."
        ],
        "things_to_avoid": [
            "DO NOT apply human eye drops or corticosteroid ointments to injured eyes.",
            "DO NOT wait until morning if your cat is straining repeatedly in the litter box without passing urine.",
            "DO NOT give human antihistamines or aspirin without professional veterinary dosing."
        ],
        "vet_recommendation": (
            "When in doubt, always err on the side of safety. Veterinary emergency teams are trained to assess and triage acute conditions."
        )
    }
]
