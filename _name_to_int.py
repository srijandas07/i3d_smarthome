def _name_to_int(name, protocol):

    if protocol == 'CS':

        integer=0
        if name=="Cook.Cleandishes":
            integer=1
        elif name=="Cook.Cleanup":
            integer=2
        elif name=="Cook.Cut":
            integer=3
        elif name=="Cook.Stir":
            integer=4
        elif name=="Cook.Usestove":
            integer=5
        elif name=="Cutbread":
            integer=6
        elif name=="Drink.Frombottle":
            integer=7
        elif name=="Drink.Fromcan":
            integer=8
        elif name=="Drink.Fromcup":
            integer=9
        elif name=="Drink.Fromglass":
            integer=10
        elif name=="Eat.Attable":
            integer=11
        elif name=="Eat.Snack":
            integer=12
        elif name=="Enter":
            integer=13
        elif name=="Getup":
            integer=14
        elif name=="Laydown":
            integer=15
        elif name=="Leave":
            integer=16
        elif name=="Makecoffee.Pourgrains":
            integer=17
        elif name=="Makecoffee.Pourwater":
            integer=18
        elif name=="Maketea.Boilwater":
            integer=19
        elif name=="Maketea.Insertteabag":
            integer=20
        elif name=="Pour.Frombottle":
            integer=21
        elif name=="Pour.Fromcan":
            integer=22
        elif name=="Pour.Fromkettle":
            integer=23
        elif name=="Readbook":
            integer=24
        elif name=="Sitdown":
            integer=25
        elif name=="Takepills":
            integer=26
        elif name=="Uselaptop":
            integer=27
        elif name=="Usetablet":
            integer=28
        elif name=="Usetelephone":
            integer=29
        elif name=="Walk":
            integer=30
        elif name=="WatchTV":
            integer=31

    else:

        integer=0
        if name=="Cutbread":
            integer=1
        elif name=="Drink.Frombottle":
            integer=2
        elif name=="Drink.Fromcan":
            integer=3
        elif name=="Drink.Fromcup":
            integer=4
        elif name=="Drink.Fromglass":
            integer=5
        elif name=="Eat.Attable":
            integer=6
        elif name=="Eat.Snack":
            integer=7
        elif name=="Enter":
            integer=8
        elif name=="Getup":
            integer=9
        elif name=="Leave":
            integer=10
        elif name=="Pour.Frombottle":
            integer=11
        elif name=="Pour.Fromcan":
            integer=12
        elif name=="Readbook":
            integer=13
        elif name=="Sitdown":
            integer=14
        elif name=="Takepills":
            integer=15
        elif name=="Uselaptop":
            integer=16
        elif name=="Usetablet":
            integer=17
        elif name=="Usetelephone":
            integer=18
        elif name=="Walk":
            integer=19
    return integer
