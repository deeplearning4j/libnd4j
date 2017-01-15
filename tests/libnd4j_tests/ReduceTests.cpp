//
// Created by agibsonccc on 1/15/17.
//
#include "testinclude.h"

    class ReduceTest : public testing::Test {
    public:
        int shape[2] = {500,3};
        float x[1500] = {4.0,2.0,3.0,8.0,4.0,6.0,12.0,6.0,9.0,16.0,8.0,12.0,20.0,10.0,15.0,24.0,12.0,18.0,28.0,14.0,21.0,32.0,16.0,24.0,36.0,18.0,27.0,40.0,20.0,30.0,44.0,22.0,33.0,48.0,24.0,36.0,52.0,26.0,39.0,56.0,28.0,42.0,60.0,30.0,45.0,64.0,32.0,48.0,68.0,34.0,51.0,72.0,36.0,54.0,76.0,38.0,57.0,80.0,40.0,60.0,84.0,42.0,63.0,88.0,44.0,66.0,92.0,46.0,69.0,96.0,48.0,72.0,100.0,50.0,75.0,104.0,52.0,78.0,108.0,54.0,81.0,112.0,56.0,84.0,116.0,58.0,87.0,120.0,60.0,90.0,124.0,62.0,93.0,128.0,64.0,96.0,132.0,66.0,99.0,136.0,68.0,102.0,140.0,70.0,105.0,144.0,72.0,108.0,148.0,74.0,111.0,152.0,76.0,114.0,156.0,78.0,117.0,160.0,80.0,120.0,164.0,82.0,123.0,168.0,84.0,126.0,172.0,86.0,129.0,176.0,88.0,132.0,180.0,90.0,135.0,184.0,92.0,138.0,188.0,94.0,141.0,192.0,96.0,144.0,196.0,98.0,147.0,200.0,100.0,150.0,204.0,102.0,153.0,208.0,104.0,156.0,212.0,106.0,159.0,216.0,108.0,162.0,220.0,110.0,165.0,224.0,112.0,168.0,228.0,114.0,171.0,232.0,116.0,174.0,236.0,118.0,177.0,240.0,120.0,180.0,244.0,122.0,183.0,248.0,124.0,186.0,252.0,126.0,189.0,256.0,128.0,192.0,260.0,130.0,195.0,264.0,132.0,198.0,268.0,134.0,201.0,272.0,136.0,204.0,276.0,138.0,207.0,280.0,140.0,210.0,284.0,142.0,213.0,288.0,144.0,216.0,292.0,146.0,219.0,296.0,148.0,222.0,300.0,150.0,225.0,304.0,152.0,228.0,308.0,154.0,231.0,312.0,156.0,234.0,316.0,158.0,237.0,320.0,160.0,240.0,324.0,162.0,243.0,328.0,164.0,246.0,332.0,166.0,249.0,336.0,168.0,252.0,340.0,170.0,255.0,344.0,172.0,258.0,348.0,174.0,261.0,352.0,176.0,264.0,356.0,178.0,267.0,360.0,180.0,270.0,364.0,182.0,273.0,368.0,184.0,276.0,372.0,186.0,279.0,376.0,188.0,282.0,380.0,190.0,285.0,384.0,192.0,288.0,388.0,194.0,291.0,392.0,196.0,294.0,396.0,198.0,297.0,400.0,200.0,300.0,404.0,202.0,303.0,408.0,204.0,306.0,412.0,206.0,309.0,416.0,208.0,312.0,420.0,210.0,315.0,424.0,212.0,318.0,428.0,214.0,321.0,432.0,216.0,324.0,436.0,218.0,327.0,440.0,220.0,330.0,444.0,222.0,333.0,448.0,224.0,336.0,452.0,226.0,339.0,456.0,228.0,342.0,460.0,230.0,345.0,464.0,232.0,348.0,468.0,234.0,351.0,472.0,236.0,354.0,476.0,238.0,357.0,480.0,240.0,360.0,484.0,242.0,363.0,488.0,244.0,366.0,492.0,246.0,369.0,496.0,248.0,372.0,500.0,250.0,375.0,504.0,252.0,378.0,508.0,254.0,381.0,512.0,256.0,384.0,516.0,258.0,387.0,520.0,260.0,390.0,524.0,262.0,393.0,528.0,264.0,396.0,532.0,266.0,399.0,536.0,268.0,402.0,540.0,270.0,405.0,544.0,272.0,408.0,548.0,274.0,411.0,552.0,276.0,414.0,556.0,278.0,417.0,560.0,280.0,420.0,564.0,282.0,423.0,568.0,284.0,426.0,572.0,286.0,429.0,576.0,288.0,432.0,580.0,290.0,435.0,584.0,292.0,438.0,588.0,294.0,441.0,592.0,296.0,444.0,596.0,298.0,447.0,600.0,300.0,450.0,604.0,302.0,453.0,608.0,304.0,456.0,612.0,306.0,459.0,616.0,308.0,462.0,620.0,310.0,465.0,624.0,312.0,468.0,628.0,314.0,471.0,632.0,316.0,474.0,636.0,318.0,477.0,640.0,320.0,480.0,644.0,322.0,483.0,648.0,324.0,486.0,652.0,326.0,489.0,656.0,328.0,492.0,660.0,330.0,495.0,664.0,332.0,498.0,668.0,334.0,501.0,672.0,336.0,504.0,676.0,338.0,507.0,680.0,340.0,510.0,684.0,342.0,513.0,688.0,344.0,516.0,692.0,346.0,519.0,696.0,348.0,522.0,700.0,350.0,525.0,704.0,352.0,528.0,708.0,354.0,531.0,712.0,356.0,534.0,716.0,358.0,537.0,720.0,360.0,540.0,724.0,362.0,543.0,728.0,364.0,546.0,732.0,366.0,549.0,736.0,368.0,552.0,740.0,370.0,555.0,744.0,372.0,558.0,748.0,374.0,561.0,752.0,376.0,564.0,756.0,378.0,567.0,760.0,380.0,570.0,764.0,382.0,573.0,768.0,384.0,576.0,772.0,386.0,579.0,776.0,388.0,582.0,780.0,390.0,585.0,784.0,392.0,588.0,788.0,394.0,591.0,792.0,396.0,594.0,796.0,398.0,597.0,800.0,400.0,600.0,804.0,402.0,603.0,808.0,404.0,606.0,812.0,406.0,609.0,816.0,408.0,612.0,820.0,410.0,615.0,824.0,412.0,618.0,828.0,414.0,621.0,832.0,416.0,624.0,836.0,418.0,627.0,840.0,420.0,630.0,844.0,422.0,633.0,848.0,424.0,636.0,852.0,426.0,639.0,856.0,428.0,642.0,860.0,430.0,645.0,864.0,432.0,648.0,868.0,434.0,651.0,872.0,436.0,654.0,876.0,438.0,657.0,880.0,440.0,660.0,884.0,442.0,663.0,888.0,444.0,666.0,892.0,446.0,669.0,896.0,448.0,672.0,900.0,450.0,675.0,904.0,452.0,678.0,908.0,454.0,681.0,912.0,456.0,684.0,916.0,458.0,687.0,920.0,460.0,690.0,924.0,462.0,693.0,928.0,464.0,696.0,932.0,466.0,699.0,936.0,468.0,702.0,940.0,470.0,705.0,944.0,472.0,708.0,948.0,474.0,711.0,952.0,476.0,714.0,956.0,478.0,717.0,960.0,480.0,720.0,964.0,482.0,723.0,968.0,484.0,726.0,972.0,486.0,729.0,976.0,488.0,732.0,980.0,490.0,735.0,984.0,492.0,738.0,988.0,494.0,741.0,992.0,496.0,744.0,996.0,498.0,747.0,1000.0,500.0,750.0,1004.0,502.0,753.0,1008.0,504.0,756.0,1012.0,506.0,759.0,1016.0,508.0,762.0,1020.0,510.0,765.0,1024.0,512.0,768.0,1028.0,514.0,771.0,1032.0,516.0,774.0,1036.0,518.0,777.0,1040.0,520.0,780.0,1044.0,522.0,783.0,1048.0,524.0,786.0,1052.0,526.0,789.0,1056.0,528.0,792.0,1060.0,530.0,795.0,1064.0,532.0,798.0,1068.0,534.0,801.0,1072.0,536.0,804.0,1076.0,538.0,807.0,1080.0,540.0,810.0,1084.0,542.0,813.0,1088.0,544.0,816.0,1092.0,546.0,819.0,1096.0,548.0,822.0,1100.0,550.0,825.0,1104.0,552.0,828.0,1108.0,554.0,831.0,1112.0,556.0,834.0,1116.0,558.0,837.0,1120.0,560.0,840.0,1124.0,562.0,843.0,1128.0,564.0,846.0,1132.0,566.0,849.0,1136.0,568.0,852.0,1140.0,570.0,855.0,1144.0,572.0,858.0,1148.0,574.0,861.0,1152.0,576.0,864.0,1156.0,578.0,867.0,1160.0,580.0,870.0,1164.0,582.0,873.0,1168.0,584.0,876.0,1172.0,586.0,879.0,1176.0,588.0,882.0,1180.0,590.0,885.0,1184.0,592.0,888.0,1188.0,594.0,891.0,1192.0,596.0,894.0,1196.0,598.0,897.0,1200.0,600.0,900.0,1204.0,602.0,903.0,1208.0,604.0,906.0,1212.0,606.0,909.0,1216.0,608.0,912.0,1220.0,610.0,915.0,1224.0,612.0,918.0,1228.0,614.0,921.0,1232.0,616.0,924.0,1236.0,618.0,927.0,1240.0,620.0,930.0,1244.0,622.0,933.0,1248.0,624.0,936.0,1252.0,626.0,939.0,1256.0,628.0,942.0,1260.0,630.0,945.0,1264.0,632.0,948.0,1268.0,634.0,951.0,1272.0,636.0,954.0,1276.0,638.0,957.0,1280.0,640.0,960.0,1284.0,642.0,963.0,1288.0,644.0,966.0,1292.0,646.0,969.0,1296.0,648.0,972.0,1300.0,650.0,975.0,1304.0,652.0,978.0,1308.0,654.0,981.0,1312.0,656.0,984.0,1316.0,658.0,987.0,1320.0,660.0,990.0,1324.0,662.0,993.0,1328.0,664.0,996.0,1332.0,666.0,999.0,1336.0,668.0,1002.0,1340.0,670.0,1005.0,1344.0,672.0,1008.0,1348.0,674.0,1011.0,1352.0,676.0,1014.0,1356.0,678.0,1017.0,1360.0,680.0,1020.0,1364.0,682.0,1023.0,1368.0,684.0,1026.0,1372.0,686.0,1029.0,1376.0,688.0,1032.0,1380.0,690.0,1035.0,1384.0,692.0,1038.0,1388.0,694.0,1041.0,1392.0,696.0,1044.0,1396.0,698.0,1047.0,1400.0,700.0,1050.0,1404.0,702.0,1053.0,1408.0,704.0,1056.0,1412.0,706.0,1059.0,1416.0,708.0,1062.0,1420.0,710.0,1065.0,1424.0,712.0,1068.0,1428.0,714.0,1071.0,1432.0,716.0,1074.0,1436.0,718.0,1077.0,1440.0,720.0,1080.0,1444.0,722.0,1083.0,1448.0,724.0,1086.0,1452.0,726.0,1089.0,1456.0,728.0,1092.0,1460.0,730.0,1095.0,1464.0,732.0,1098.0,1468.0,734.0,1101.0,1472.0,736.0,1104.0,1476.0,738.0,1107.0,1480.0,740.0,1110.0,1484.0,742.0,1113.0,1488.0,744.0,1116.0,1492.0,746.0,1119.0,1496.0,748.0,1122.0,1500.0,750.0,1125.0,1504.0,752.0,1128.0,1508.0,754.0,1131.0,1512.0,756.0,1134.0,1516.0,758.0,1137.0,1520.0,760.0,1140.0,1524.0,762.0,1143.0,1528.0,764.0,1146.0,1532.0,766.0,1149.0,1536.0,768.0,1152.0,1540.0,770.0,1155.0,1544.0,772.0,1158.0,1548.0,774.0,1161.0,1552.0,776.0,1164.0,1556.0,778.0,1167.0,1560.0,780.0,1170.0,1564.0,782.0,1173.0,1568.0,784.0,1176.0,1572.0,786.0,1179.0,1576.0,788.0,1182.0,1580.0,790.0,1185.0,1584.0,792.0,1188.0,1588.0,794.0,1191.0,1592.0,796.0,1194.0,1596.0,798.0,1197.0,1600.0,800.0,1200.0,1604.0,802.0,1203.0,1608.0,804.0,1206.0,1612.0,806.0,1209.0,1616.0,808.0,1212.0,1620.0,810.0,1215.0,1624.0,812.0,1218.0,1628.0,814.0,1221.0,1632.0,816.0,1224.0,1636.0,818.0,1227.0,1640.0,820.0,1230.0,1644.0,822.0,1233.0,1648.0,824.0,1236.0,1652.0,826.0,1239.0,1656.0,828.0,1242.0,1660.0,830.0,1245.0,1664.0,832.0,1248.0,1668.0,834.0,1251.0,1672.0,836.0,1254.0,1676.0,838.0,1257.0,1680.0,840.0,1260.0,1684.0,842.0,1263.0,1688.0,844.0,1266.0,1692.0,846.0,1269.0,1696.0,848.0,1272.0,1700.0,850.0,1275.0,1704.0,852.0,1278.0,1708.0,854.0,1281.0,1712.0,856.0,1284.0,1716.0,858.0,1287.0,1720.0,860.0,1290.0,1724.0,862.0,1293.0,1728.0,864.0,1296.0,1732.0,866.0,1299.0,1736.0,868.0,1302.0,1740.0,870.0,1305.0,1744.0,872.0,1308.0,1748.0,874.0,1311.0,1752.0,876.0,1314.0,1756.0,878.0,1317.0,1760.0,880.0,1320.0,1764.0,882.0,1323.0,1768.0,884.0,1326.0,1772.0,886.0,1329.0,1776.0,888.0,1332.0,1780.0,890.0,1335.0,1784.0,892.0,1338.0,1788.0,894.0,1341.0,1792.0,896.0,1344.0,1796.0,898.0,1347.0,1800.0,900.0,1350.0,1804.0,902.0,1353.0,1808.0,904.0,1356.0,1812.0,906.0,1359.0,1816.0,908.0,1362.0,1820.0,910.0,1365.0,1824.0,912.0,1368.0,1828.0,914.0,1371.0,1832.0,916.0,1374.0,1836.0,918.0,1377.0,1840.0,920.0,1380.0,1844.0,922.0,1383.0,1848.0,924.0,1386.0,1852.0,926.0,1389.0,1856.0,928.0,1392.0,1860.0,930.0,1395.0,1864.0,932.0,1398.0,1868.0,934.0,1401.0,1872.0,936.0,1404.0,1876.0,938.0,1407.0,1880.0,940.0,1410.0,1884.0,942.0,1413.0,1888.0,944.0,1416.0,1892.0,946.0,1419.0,1896.0,948.0,1422.0,1900.0,950.0,1425.0,1904.0,952.0,1428.0,1908.0,954.0,1431.0,1912.0,956.0,1434.0,1916.0,958.0,1437.0,1920.0,960.0,1440.0,1924.0,962.0,1443.0,1928.0,964.0,1446.0,1932.0,966.0,1449.0,1936.0,968.0,1452.0,1940.0,970.0,1455.0,1944.0,972.0,1458.0,1948.0,974.0,1461.0,1952.0,976.0,1464.0,1956.0,978.0,1467.0,1960.0,980.0,1470.0,1964.0,982.0,1473.0,1968.0,984.0,1476.0,1972.0,986.0,1479.0,1976.0,988.0,1482.0,1980.0,990.0,1485.0,1984.0,992.0,1488.0,1988.0,994.0,1491.0,1992.0,996.0,1494.0,1996.0,998.0,1497.0,2000.0,1000.0,1500.0};
        float result[1500] = {0};
        int dimension[1] = {0};
        int dimensionLength = 1;
        float theoreticalMin[3] = {4,2,3};
        float theoreticalMax[3] = {2000.00, 1000.00, 1500.00};
        float theoreticalRange[3] = {1996.00, 998.00, 1497.00};
    };



TEST_F(ReduceTest,MatrixTest) {
    int opNum = 4;
    int *xShapeInfo = shape::shapeBuffer(2,shape);
    int *resultShapeInfo = shape::computeResultShape(xShapeInfo,dimension,dimensionLength);
    int resultLengthAssertion = 3;
    ASSERT_EQ(resultLengthAssertion,shape::length(resultShapeInfo));
    shape::TAD *tad = new shape::TAD(xShapeInfo,dimension,dimensionLength);
    float none[1] = {0};
    tad->createTadOnlyShapeInfo();
    tad->createOffsets();
    int tadElementWiseStride = shape::elementWiseStride(tad->tadOnlyShapeInfo);
    ASSERT_EQ(3,tadElementWiseStride);
    functions::reduce::ReduceFunction<float>::exec(
            opNum,
            x,
            xShapeInfo,
            none,
            result,
            resultShapeInfo,
            dimension,
            dimensionLength,
            tad->tadOnlyShapeInfo,
            tad->tadOffsets);

    for(int i = 0; i < shape::length(resultShapeInfo); i++)
        printf("%f\n",result[i]);

    delete[] resultShapeInfo;
    delete tad;
    delete[] xShapeInfo;
}