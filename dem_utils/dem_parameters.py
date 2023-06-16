import numpy as np
import math
from skimage import io as skio


__D8_DIRECTIONS__ = {
    1: (-1, 1),  # northeast
    2: (0, 1),  # east
    4: (1, 1),  # southeast
    8: (1, 0),  # south
    16: (1, -1),  # southwest
    32: (0, -1),  # west
    64: (-1, -1),  # northwest
    128: (-1, 0)  # north
}

__neighbor_D8_DIRECTIONS__ = {  # 如果对应的邻居流向是这个方向，才是流向中心cell
    0: (16, -1, 1),  # northeast
    1: (32, 0, 1),  # east
    2: (64, 1, 1),  # southeast
    3: (128, 1, 0),  # south
    4: (1, 1, -1),  # northwest
    5: (2, 0, -1),  # west
    6: (4, -1, -1),  # northwest
    7: (8, -1, 0)  # north
}


def accumation(dem):
    row = dem.shape[0]
    column = dem.shape[1]
    dir_dem = np.zeros(shape=(row, column), dtype=np.float64)
    for i in range(row):
        for j in range(column):
            if i == 0:
                dir_dem[i][j] = 128
            elif i == row - 1:
                dir_dem[i][j] = 8
            elif j == 0:
                dir_dem[i][j] = 32
            elif j == column - 1:
                dir_dem[i][j] = 2
            else:
                neighbors = []
                drop = []
                for k in range(8):
                    coord = __D8_DIRECTIONS__[math.pow(2, k)]
                    if i + coord[0] >= 0 and i + coord[0] < row and j + coord[1] >= 0 and j + coord[1] < column:
                        neighbors.append((coord[0], coord[1], dem[i + coord[0]][j + coord[1]].item()))
                        drop_temp = dem[i][j].item() - neighbors[k][2]
                        if coord[0] == 0 or coord[1] == 0:
                            drop_temp = drop_temp
                        else:
                            drop_temp = drop_temp / math.sqrt(2)
                        drop.append((drop_temp, k))
                drop = sorted(drop, key=lambda drop: drop[0], reverse=True)
                direction = 0
                if (drop[0][0] > 0):
                    direction = math.pow(2, drop[0][1])
                else:
                    direction = -1
                dir_dem[i][j] = direction
    acc_dem = np.zeros(shape=(row, column), dtype=np.float64)
    for i in range(row):
        for j in range(column):
            temp_acc = 0
            flows = [(i, j)]
            k = 0
            while k != len(flows):
                center = [flows[k][0], flows[k][1]]
                for l in range(8):
                    coord = __neighbor_D8_DIRECTIONS__[l]
                    if center[0] + coord[1] >= 0 and center[0] + coord[1] < row and center[1] + coord[2] >= 0 and \
                            center[1] + coord[2] < column:
                        if dir_dem[center[0] + coord[1]][center[1] + coord[2]] == coord[0]:
                            if acc_dem[center[0] + coord[1]][center[1] + coord[2]] != 0:
                                temp_acc = temp_acc + acc_dem[center[0] + coord[1]][center[1] + coord[2]] + 1
                            else:
                                flows.append((center[0] + coord[1], center[1] + coord[2]))
                                temp_acc = temp_acc + 1
                k = k + 1
            acc_dem[i][j] = temp_acc

    # _, thresholded_river = cv2.threshold(acc_dem, 50, 1, cv2.THRESH_BINARY)
    # return thresholded_river
    #return acc_dem / (row * column)
    return acc_dem


def aspect(dem, dem_size):
    row = dem.shape[0]
    column = dem.shape[1]
    aspect = np.zeros(shape=(row, column), dtype=np.float64)
    for i in range(row):
        for j in range(column):
            neighbors = []
            for k in range(8):
                coord = __D8_DIRECTIONS__[math.pow(2, k)]
                if i + coord[0] >= 0 and i + coord[0] < row and j + coord[1] >= 0 and j + coord[1] < column:
                    neighbors.append(dem[i + coord[0]][j + coord[1]].item())
                else:
                    neighbors.append(dem[i][j])
            dx = ((neighbors[0] + 2 * neighbors[1] + neighbors[2]) - (
                    neighbors[6] + neighbors[5] * 2 + neighbors[4])) / (8 * dem_size)
            dy = -(((neighbors[0] + 2 * neighbors[7] + neighbors[6]) - (
                    neighbors[2] + 2 * neighbors[3] + neighbors[4])) / (8 * dem_size))
            ''' Changed
            Aspect_rad = 0
            if dx != 0:
                Aspect_rad = math.atan2(dy, -dx)
                if Aspect_rad < 0:
                    Aspect_rad = 2 * math.pi + Aspect_rad
            if dx == 0:
                if dy > 0:
                    Aspect_rad = math.pi / 2
                elif dy < 0:
                    Aspect_rad = 2 * math.pi - math.pi / 2
                else:
                    Aspect_rad = Aspect_rad
            aspect[i][j] = Aspect_rad
            '''

            # from https://desktop.arcgis.com/en/arcmap/10.3/tools/spatial-analyst-toolbox/how-aspect-works.htm
            Aspect_degree = math.atan2(dy, -dx)*180/math.pi
            if Aspect_degree < 0:
                Aspect_degree = 90 - Aspect_degree
            elif Aspect_degree>90:
                Aspect_degree = 360 - Aspect_degree + 90
            else:
                Aspect_degree = 90 - Aspect_degree
            aspect[i][j] = Aspect_degree
    # return aspect / (2 * math.pi)
    return aspect


def slope(dem, dem_size):
    row = dem.shape[0]
    column = dem.shape[1]
    dem_slope = np.zeros(shape=(row, column), dtype=np.float64)
    for i in range(row):
        for j in range(column):
            neighbors = []
            for k in range(8):
                coord = __D8_DIRECTIONS__[math.pow(2, k)]
                if i + coord[0] >= 0 and i + coord[0] < row and j + coord[1] >= 0 and j + coord[1] < column:
                    neighbors.append(dem[i + coord[0]][j + coord[1]].item())
                else:
                    neighbors.append(dem[i][j])
            dx = ((neighbors[0] + 2 * neighbors[1] + neighbors[2]) - (
                    neighbors[6] + neighbors[5] * 2 + neighbors[4])) / (8 * dem_size)
            dy = -((neighbors[0] + 2 * neighbors[7] + neighbors[6]) - (
                    neighbors[2] + 2 * neighbors[3] + neighbors[4])) / (8 * dem_size)
            ij_slope = math.sqrt(dx * dx + dy * dy)
            ij_slope = math.atan(ij_slope) * 180 / math.pi
            dem_slope[i][j] = ij_slope
    #return dem_slope / 360.0
    return dem_slope


def hillshade(dem, dem_size, Altitude=45, Azimuth=315):
    zenith_rad = (90 - Altitude) * math.pi / 180.0
    asimuth_math = 360 - Azimuth + 90
    if asimuth_math >= 360:
        asimuth_math = asimuth_math - 360
    asimuth_rad = asimuth_math * math.pi / 180.0
    row = dem.shape[0]
    column = dem.shape[1]
    dem_hillshade = np.zeros(shape=(row, column), dtype=np.float64)
    for i in range(row):
        for j in range(column):
            neighbors = []
            for k in range(8):
                coord = __D8_DIRECTIONS__[math.pow(2, k)]
                if i + coord[0] >= 0 and i + coord[0] < row and j + coord[1] >= 0 and j + coord[1] < column:
                    neighbors.append(dem[i + coord[0]][j + coord[1]].item())
                else:
                    neighbors.append(dem[i][j])
            dx = ((neighbors[0] + 2 * neighbors[1] + neighbors[2]) - (
                    neighbors[6] + neighbors[5] * 2 + neighbors[4])) / (8 * dem_size)
            dy = -(((neighbors[0] + 2 * neighbors[7] + neighbors[6]) - (
                    neighbors[2] + 2 * neighbors[3] + neighbors[4])) / (8 * dem_size))
            ij_slope = math.sqrt(dx * dx + dy * dy)
            ij_slope = math.atan(ij_slope)
            Aspect_rad = 0
            if dx != 0:
                Aspect_rad = math.atan2(dy, -dx)
                if Aspect_rad < 0:
                    Aspect_rad = 2 * math.pi + Aspect_rad
            if dx == 0:
                if dy > 0:
                    Aspect_rad = math.pi / 2
                elif dy < 0:
                    Aspect_rad = 2 * math.pi - math.pi / 2
                else:
                    Aspect_rad = Aspect_rad

            hillshade_ij = 255.0 * ((math.cos(zenith_rad) * math.cos(ij_slope)) + (
                    math.sin(zenith_rad) * math.sin(ij_slope) * math.cos(asimuth_rad - Aspect_rad)))
            if (hillshade_ij < 0):
                hillshade_ij = 0
            dem_hillshade[i][j] = hillshade_ij
    return dem_hillshade / 255.0


def curvature(dem, dem_size):
    row = dem.shape[0]
    column = dem.shape[1]
    dem_curvature = np.zeros(shape=(row, column), dtype=np.float64)
    for i in range(row):
        for j in range(column):
            neighbors = []
            for k in range(8):
                coord = __D8_DIRECTIONS__[math.pow(2, k)]
                if i + coord[0] >= 0 and i + coord[0] < row and j + coord[1] >= 0 and j + coord[1] < column:
                    neighbors.append(dem[i + coord[0]][j + coord[1]].item())
                else:
                    neighbors.append(dem[i][j])
            D = ((neighbors[5] + neighbors[1]) / 2.0 - dem[i][j]) / (dem_size * dem_size)
            E = ((neighbors[7] + neighbors[3]) / 2.0 - dem[i][j]) / (dem_size * dem_size)
            ij_curvature = -2 * (D + E)  # * 100
            dem_curvature[i][j] = ij_curvature
    return dem_curvature


def RMSE(dem1, dem2):
    row = dem1.shape[1]
    column = dem1.shape[2]
    rmse = 0
    for i in range(row):
        for j in range(column):
            rmse += math.pow((dem1[0][i][j] - dem2[0][i][j]), 2)
    rmse = math.sqrt(rmse / (row * column))
    return rmse


def a_file(filepath, target_tensor):
    with open(filepath, 'a') as af:
        num_rows, num_cols = target_tensor.shape
        for i in range(num_rows):
            for j in range(num_cols):
                af.write(str(target_tensor[i][j].item()) + ',')
            af.write('\n')



def getdis(img, fake):
    np_img = img.numpy()
    np_fake = fake.numpy()
    one_img = np_img.flatten()
    one_fake = np_fake.flatten()
    size = len(one_img)
    size = math.sqrt(size)
    img_loc = np.argsort(one_img)
    fake_loc = np.argsort(one_fake)
    # max-20
    m_img_loc_xy = np.zeros((20, 2))
    m_fake_loc_xy = np.zeros((20, 2))
    j = 0
    for i in range(len(img_loc) - 20, len(img_loc)):
        m_img_loc_xy[j][0] = int(img_loc[i] / size)
        m_img_loc_xy[j][1] = img_loc[i] % size
        m_fake_loc_xy[j][0] = int(fake_loc[i] / size)
        m_fake_loc_xy[j][1] = fake_loc[i] % size
        j = j + 1
    # min-20
    n_img_loc_xy = np.zeros((20, 2))
    n_fake_loc_xy = np.zeros((20, 2))
    for i in range(20):
        n_img_loc_xy[i][0] = int(img_loc[i] / size)
        n_img_loc_xy[i][1] = img_loc[i] % size
        n_fake_loc_xy[i][0] = int(fake_loc[i] / size)
        n_fake_loc_xy[i][1] = fake_loc[i] % size
    # dis
    max_dis = 0
    min_dis = 0
    for i in range(20):
        max_dis += math.sqrt(
            math.pow(m_img_loc_xy[i][0] - m_fake_loc_xy[i][0], 2) + math.pow(m_img_loc_xy[i][1] - m_fake_loc_xy[i][1],
                                                                             2))
        min_dis += math.sqrt(
            math.pow(n_img_loc_xy[i][0] - n_fake_loc_xy[i][0], 2) + math.pow(n_img_loc_xy[i][1] - n_fake_loc_xy[i][1],
                                                                             2))
    max_dis = max_dis / 20
    min_dis = min_dis / 20
    return max_dis, min_dis


def cmp_DEMParameter(
    demA_file, demB_file, cell_size, scale=None
):
    demA = skio.imread(demA_file)
    demB = skio.imread(demB_file)

    if scale:
        demA = demA[scale:-scale, scale:-scale]
        demB = demB[scale:-scale, scale:-scale]
    
    aspect_diff = aspect(demA, cell_size) - aspect(demB, cell_size)
    slope_diff = slope(demA, cell_size) - slope(demB, cell_size)

    return {
        'aspect_mae': np.abs(aspect_diff).mean(),
        'slope_mae': np.abs(slope_diff).mean(),
        'aspect_rmse': np.sqrt(np.square(aspect_diff).sum()),
        'slope_rmse': np.sqrt(np.square(slope_diff).sum()),
    }
