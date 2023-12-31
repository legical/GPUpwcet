import sys
import os
# filename = "appSortedReuse"
# gridDim = [4,1,1]
# print("???", file=sys.stderr)
if len(sys.argv) < 5:
    print("Error Input!")
    exit(1)
filename = sys.argv[1]
gridDim = [int(x) for x in sys.argv[2:]]

minTag = None
maxTag = 0
current_tag = None

count = 0
used_time = 0
# warp_dict = dict()
# tag_dict = dict()
current_warp_list = set()
with open(filename) as f:
    for line in f:
        # print(count)
        count += 1
        strs = line.split(" ")
        blockId = int(strs[1])
        warpId = int(strs[2])
        smId = int(strs[3])
        tag = int(strs[4])
        if(current_tag != tag):
            if(current_tag is not None):
                print(str(current_tag)+","+str(used_time - 1)+",",list(current_warp_list))
            current_tag = tag
            current_warp_list = set()
            used_time = 0
        used_time += 1
        if minTag is None or minTag > tag:
            minTag = tag
        if maxTag is None or maxTag < tag:
            maxTag = tag
        current_warp_list.add((blockId, warpId, smId))
        # if((blockId, warpId, smId) not in warp_dict):
        #     warp_dict[(blockId, warpId, smId)] = []
        # warp_dict[(blockId, warpId, smId)].append(tag)
        # if(tag not in tag_dict):
            # tag_dict[tag] = 0
        # if strs[0] == 'Reuse':
        #     reused_time += 1
print(str(current_tag)+","+str(used_time - 1)+",",current_warp_list)
print("Working set size: {} bytes".format((maxTag-minTag + 1) * 128), file=sys.stderr)
print("Total sector queries: {}".format(count), file=sys.stderr)
print("Minimal line tag: {}".format(minTag), file=sys.stderr)
print("Maximal line tag: {}".format(maxTag), file=sys.stderr)
# line_dict = dict()
# for tag in range(minTag, maxTag + 1):
#     if(tag not in line_dict):
#         line_dict[tag] = []
    # all_warps = []
    # reused = 0
    # if tag in tag_dict:
    #     reused = tag_dict[tag]
    # for k in warp_dict:
    #     if tag in warp_dict[k]:
    #         all_warps.append(k)
    # print(str(reused)+",",all_warps)