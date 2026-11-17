"""Class hierarchy definition. Edit this file to change the dashboard's hierarchy.

Same layout as the training code: for each level, a list of group names
(coarse_names) and, in the same order, the classes that belong to each group
(groups). The lists at the bottom run from the finest level to the coarsest.
"""

# --- Dashboard-only settings (not used by the training code) -----------------
ROOT = "Zooplankton"   # label for the top of the sunburst/treemap
COLOR_LEVEL = 2        # bar-chart "Branch" colors use this level (1 = coarsest)

# --- Hierarchy ----------------------------------------------------------------
# 2.Create hierarchical dataset

LEVELS = 3

coarse_names1 = ['Zoop-yes', 'Zoop-No']
groups1 = [
    [
        'Copepoda','Cladocera','Bosminidae', 'Bosmina_1','Eubosmina','Daphnia',
        'Cyclopoid','Harpacticoid','Holopedium','Calanoid','Rotifer',
        'Nauplius_Copepod','Trichocerca','Asplanchna', 'Kellicottia',
        'Conochilus', 'Chydorus'

    ],
    
    [
        'Debris','Bubbles','Exoskeleton','Fiber_Squiggly', 
        'Fiber_Hairlike','Plant_Matter'
    ]
]

coarse_names2 = [
    'Copepoda', 'Cladocera','Rotifer','Bubbles', 'Exoskeleton',
    'Fiber', 'Plant_Matter'
]
groups2 = [
    ['Copepoda','Cyclopoid','Calanoid','Harpacticoid','Nauplius_Copepod'],
    ['Cladocera','Bosminidae','Bosmina_1','Eubosmina','Daphnia','Holopedium','Chydorus'],
    ['Rotifer','Trichocerca','Asplanchna','Conochilus','Kellicottia'],
    ['Bubbles'],
    ['Exoskeleton'],
    ['Fiber_Squiggly','Fiber_Hairlike'],
    ['Plant_Matter']
]

coarse_names3 = [
    'Cyclopoid','Calanoid','Harpacticoid','Nauplius_Copepod',
    'Bosmina','Holopedium','Daphnia','Chydorus',
    'Kellicottia','Conochilus','Asplanchna','Trichocerca',
    'Bubbles','Exoskeleton','Fiber','Plant_Matter'
]
groups3 = [
    ['Cyclopoid'],
    ['Calanoid'],
    ['Harpacticoid'],
    ['Nauplius_Copepod'],
    ['Bosminidae','Bosmina_1','Eubosmina'],
    ['Holopedium'],
    ['Daphnia'],
    ['Chydorus'],
    ['Kellicottia'],
    ['Conochilus'],
    ['Asplanchna'],
    ['Trichocerca'],
    ['Bubbles'],
    ['Exoskeleton'],
    ['Fiber_Squiggly','Fiber_Hairlike'],
    ['Plant_Matter'] 
] 
coarse_names = [coarse_names3,coarse_names2,coarse_names1]
groups = [groups3, groups2, groups1]
