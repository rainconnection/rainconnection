import numpy as np

# 기존 검색된 소스코드 : 단일 값 비교 및 n:1, n:n 비교 가능
def haversine_np(lat1, lon1, lat2, lon2):
    #- input : two points
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0008 * c
    return km

# 약간 변형 n:m 비교 가능, matrix 형태로 출
def haversine_np_arrays(lat1, lon1, lat2, lon2):
    #- input : two array
    #- output : NxM matrix, N = length of array of points1, M = length of array of points2
    N = len(lat1)
    lat1 = lat1.reshape(N,1)
    lon1 = lon1.reshape(N,1)
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0008 * c #지구 반지
    return km 