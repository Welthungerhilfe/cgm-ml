""" original list
BLACKLIST_QRCODES = [
    "1585000019-syglokl9nx",  # only to test (part of mini)
    "1585366118-qao4zsk0m3",  # in anon-depthmap-95k, child_height = 12.7, scans/1585366118-qao4zsk0m3/102/pc_1585366118-qao4zsk0m3_1593021766372_102_026.p'  # noqa: E501
    "1585360775-fa64muouel",  # in anon-depthmap-95k, child_height = 7.9, scans/1585360775-fa64muouel/202/pc_1585360775-fa64muouel_1597205960827_202_002.p',  # noqa: E501
    '1583855791-ldfc59ywg5',  # in anon-depthmap-95k, child_height
    '1583997882-3jqstr1119',  # in anon-depthmap-95k, child_height
    '1584998372-d85ogmqucw',  # in anon-depthmap-95k, child_height
    '1585274424-3oqa4i262a',  # in anon-depthmap-95k, child_height
    '1585010027-xb21f31tvj',  # in anon-depthmap-95k, pixel_value_max = 714286.0, b'scans/1585010027-xb21f31tvj/101/pc_1585010027-xb21f31tvj_1592674994326_101_015.p'  # noqa: E501
] + ['1583855791-ldfc59ywg5', '1583997882-3jqstr1119', '1584998372-d85ogmqucw', '1585274424-3oqa4i262a', '1585360775-fa64muouel', '1585366118-qao4zsk0m3', '1583464618-et67nim5pv', '1584997150-17hbdbbvst', '1584998076-071u80jhgb', '1585003269-r6u05tzkj7', '1585008564-6b2vy7vcbe', '1585268854-t9dc711wl2', '1585271708-jc2k2d03jd', '1585273422-lr31sn53h1', '1585273699-0o50bij8vh', '1585353338-boa4phzps5', '1585366271-ok8vyfgcdw', '1585366660-6cqmyjhmxu', '1583710796-p68osilppn', '1583831904-13qoxucgul', '1584751648-u50v4ryc54', '1584751846-e0ro1v86r2', '1584995460-ewet5gjfer', '1584995539-iry2pwtgjy', '1584996738-hyxlj4cees', '1584998005-c5ddnj0vuc', '1585001723-az25md2fir', '1585001805-bv723pvcy0', '1585005264-o4vjkmj3o5', '1585011032-l23rao0l8r', '1585274328-7g4vj56tk8', '1585355880-o0kqvrzncj', '1585361240-nievyiqbdv', '1585357787-ko2s9g4cnz', '1585014333-santnthviw', '1585014363-9iji5fk3z0', '1585014369-pohzlgwmj5', '1585014399-vypf7yka8r', '1585014410-ulsr4y63ej', '1585014460-d8a367uh2c', '1585015663-47cp0jvb3i', '1585016734-7v4g0jd834', '1585119329-nrn8xkjv8h', '1585271214-8brq5raxu9', '1585271608-y4lv0m7x58', '1585273215-crhmfcp509', '1585273459-r4dr4i318f', '1585273477-nxdc8o8bfi', '1585273510-deqcmjslb5', '1585273542-bagoat980u', '1585273581-kkzed7slcb', '1585273724-u6ofjhfyfu', '1585294285-jkx96y7ojm', '1585348323-y0pjzbft4j', '1585350176-71ux4sljgl', '1585351186-gu3jkafr22', '1585351209-vpbt1ravfb', '1585351282-kauyajspjw', '1585355756-o00y4j5eod', '1585356456-gbib6dyc0k', '1585358073-lhg8oycx8w', '1585358389-iqmjzadrg8', '1585360528-dty9a6gd3a', '1585362557-lnbh5rz6qv', '1585362994-05a8dzt7db', '1585365927-r2gdowfaus', '1583462523-zolqj1h1so', '1583462542-nvwrue6sak', '1583850151-7lqggqjbfz', '1583997887-kjl2onswvl', '1583998466-92e41regoe', '1584921649-y7mb71gtq5', '1584993440-jt1q7jvcxk', '1584995840-yc5e340qkf', '1584996554-5hc50kbv5k', '1584996982-jw4772tp4t', '1584999538-fwlijzba3n', '1585004714-vvwdktbjr4', '1585011244-en5qxroh2d', '1598211458-e3wx4hygnq', '1597609986-gdgugh9uu5', '1597814521-3wg8as3col', '1598117850-v2o0qhdbju', '1598278297-c1h5atpk76', '1584997869-2f4en811rc', '1585016701-1jny2rideq', '1597602232-8y2xmw4jxh', '1597612326-3jwkp3hqm4', '1597628974-gnhqmt2lc9', '1597929175-s2xa3r6qxb', '1597959201-cdh3nzurz9', '1598004797-25ev2qs0tp', '1598079550-hzdmi3o7ko', '1598103263-yen7e381r9', '1598150765-1d0jctea8v', '1598182099-cyc43kdirx', '1598261064-nuwoyox9av', '1598341800-7kao2jnlbf']  # noqa: E501
BLACKLIST_QRCODES = list(set(BLACKLIST_QRCODES))"""


BLACKLIST_QRCODES_8SAMPLES = [
    "1585000019-syglokl9nx",  # only to test (part of mini)
    "1585366118-qao4zsk0m3",  # in anon-depthmap-95k, child_height = 12.7, scans/1585366118-qao4zsk0m3/102/pc_1585366118-qao4zsk0m3_1593021766372_102_026.p'  # noqa: E501
    "1585360775-fa64muouel",  # in anon-depthmap-95k, child_height = 7.9, scans/1585360775-fa64muouel/202/pc_1585360775-fa64muouel_1597205960827_202_002.p',  # noqa: E501
    '1583855791-ldfc59ywg5',  # in anon-depthmap-95k, child_height
    '1583997882-3jqstr1119',  # in anon-depthmap-95k, child_height
    '1584998372-d85ogmqucw',  # in anon-depthmap-95k, child_height
    '1585274424-3oqa4i262a',  # in anon-depthmap-95k, child_height
    '1585010027-xb21f31tvj',  # in anon-depthmap-95k, pixel_value_max = 714286.0,
]


# 4x correct size (21), 4 wrong size
BLACKLIST_QRCODES_WRONG_SIZE = [
    "1585000019-syglokl9nx",  # only to test (part of mini)
    "1585366118-qao4zsk0m3",  # in anon-depthmap-95k, child_height = 12.7, scans/1585366118-qao4zsk0m3/102/pc_1585366118-qao4zsk0m3_1593021766372_102_026.p'  # noqa: E501
    "1585360775-fa64muouelXXXX",  # in anon-depthmap-95k, child_height = 7.9, scans/1585360775-fa64muouel/202/pc_1585360775-fa64muouel_1597205960827_202_002.p',  # noqa: E501
    '1583855791-ldfc59',  # in anon-depthmap-95k, child_height
    '1583997882-3jqstr1119XXXX',  # in anon-depthmap-95k, child_height
    '1584998372',  # in anon-depthmap-95k, child_height
    '1585274424-3oqa4i262a',  # in anon-depthmap-95k, child_height
    '1585010027-xb21f31tvj',  # in anon-depthmap-95k, pixel_value_max = 714286.0,
]


BLACKLIST_QRCODES_EMPTY = []
