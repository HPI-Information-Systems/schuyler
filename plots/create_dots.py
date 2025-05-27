import math

###############################################################################
# 1) Arrange n points in a 'filled circle' around (0,0)
###############################################################################
def arrange_points_filled_circle(n_points, ring_radius=1.0, spacing=1.0):
    """
    Arranges n_points in a 'filled circle' pattern around (0,0):
      - 1 point at the center if n_points >= 1
      - Then concentric rings outward:
          Ring i has radius = i * ring_radius,
          and can hold up to floor(circumference / spacing) points,
          where circumference = 2 pi * radius.

    Returns a list of (x, y) coords (centered at (0,0)) for the n_points.
    """
    coords = []
    if n_points <= 0:
        return coords
    
    # 1) Center point
    coords.append((0.0, 0.0))
    leftover = n_points - 1

    ring_i = 1
    while leftover > 0:
        radius = ring_i * ring_radius
        circumference = 2.0 * math.pi * radius
        ring_capacity = int(circumference // spacing)  # approximate max points by spacing
        ring_capacity = max(ring_capacity, 1)          # at least 1 point if circumference < spacing

        ring_count = min(ring_capacity, leftover)
        if ring_count > 0:
            angle_step = 2.0 * math.pi / ring_count
            for j in range(ring_count):
                angle = j * angle_step
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                coords.append((x, y))
        leftover -= ring_count
        ring_i += 1
    
    return coords


###############################################################################
# 2) Place each cluster on a large circle, then fill it with the above arrangement
###############################################################################
def assign_clusters_and_positions_filledcircle(
    clusters,
    big_radius=10.0,
    ring_radius=1.0,
    spacing=1.0,
    cluster_colors=None
):
    """
    clusters : list of lists
        Each sub-list is one cluster of tables, e.g.: [["T1","T2"], ["T3","T4","T5"]]
    big_radius : float
        Radius for placing cluster centers on a big circle
    ring_radius : float
        Distance between rings inside each cluster's 'filled circle'
    spacing : float
        Arc length spacing between adjacent points on the same ring
    cluster_colors : dict or None
        If given, maps cluster_index -> color string.
        If None, we auto-generate a color for each cluster index.

    Returns
    -------
    positions : dict  {table_name: {"x": float, "y": float, "color": str}}
    """
    k = len(clusters)
    positions = {}

    # 2.1) If no colors are specified, auto-assign from a palette
    default_color_list = [
        "red", "blue", "green", "magenta", "cyan",
        "orange", "purple", "brown", "teal", "violet", "olive"
    ]
    if cluster_colors is None:
        cluster_colors = {}
        for i in range(k):
            cluster_colors[i] = default_color_list[i % len(default_color_list)]
    
    # 2.2) Compute cluster centers around (0,0) on a large circle
    #      center_i = (big_radius*cos(2 pi i/k), big_radius*sin(2 pi i/k))
    cluster_centers = {}
    for i in range(k):
        angle = 2.0 * math.pi * i / k
        cx = big_radius * math.cos(angle)
        cy = big_radius * math.sin(angle)
        cluster_centers[i] = (cx, cy)

    # 2.3) Fill each cluster
    for cluster_idx, tables_in_cluster in enumerate(clusters):
        n_points = len(tables_in_cluster)
        local_coords = arrange_points_filled_circle(
            n_points, ring_radius=ring_radius, spacing=spacing
        )
        center_x, center_y = cluster_centers[cluster_idx]
        color = cluster_colors[cluster_idx]

        # offset local coords by the cluster center
        for tbl, (lx, ly) in zip(tables_in_cluster, local_coords):
            x = center_x + lx
            y = center_y + ly
            positions[tbl] = {"x": x, "y": y, "color": color}
    
    return positions


###############################################################################
# 3) Normalization (OPTIONAL)
###############################################################################
def normalize_coordinates(positions, x_range=(-1, 1), y_range=(-1, 1)):
    """
    Rescales all (x, y) in `positions` to fit in x_range x y_range.
    Useful for consistent plotting size in LaTeX.
    """
    if not positions:
        return positions

    all_x = [info["x"] for info in positions.values()]
    all_y = [info["y"] for info in positions.values()]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    if x_min == x_max or y_min == y_max:
        return positions  # no spread or only 1 point

    x_tmin, x_tmax = x_range
    y_tmin, y_tmax = y_range

    for tbl, info in positions.items():
        old_x = info["x"]
        old_y = info["y"]
        new_x = x_tmin + (old_x - x_min) * (x_tmax - x_tmin) / (x_max - x_min)
        new_y = y_tmin + (old_y - y_min) * (y_tmax - y_tmin) / (y_max - y_min)
        info["x"], info["y"] = new_x, new_y

    return positions


###############################################################################
# 4) MAIN: Combine Ground-Truth + Created Clustering
###############################################################################
if __name__ == "__main__":
    #
    # Example input:
    #
    # The *same* tables appear in both clusterings:
    #
    # ground_truth_clusters = [
    #     ["broker", "trade_request", "trade", "trade_type", "charge", "settlement", "commission_rate", "cash_transaction", "trade_history"],
    #     ["watch_item", "taxrate", "watch_list", "customer_account", "customer_taxrate", "customer", "holding", "holding_summary", "holding_history", "account_permission"],
    #     ["company_competitor", "financial", "company", "industry", "sector", "exchange", "news_item", "news_xref", "last_trade", "daily_market", "security"]
    # ]
    
    # created_clusters = [['account_permission', 'broker', 'customer', 'customer_account', 'financial', 'holding_summary'], ['customer_taxrate', 'taxrate'], ['company', 'company_competitor', 'industry', 'sector'], ['daily_market', 'last_trade', 'security'], ['news_item', 'news_xref'], ['cash_transaction', 'holding', 'holding_history', 'settlement', 'trade', 'trade_history'], ['charge', 'commission_rate', 'exchange', 'trade_request', 'trade_type'], ['watch_item', 'watch_list']]
    ground_truth_clusters = [["Badges", "Users"], 
                ["PostNotices", "PostNoticeTypes", "Posts", "PostTypes", "PostLinks", "Comments"],["VoteTypes", "Votes", "SuggestedEdits", "SuggestedEditVotes", "PostFeedback"], ["CloseReasonTypes", "CloseAsOffTopicReasonTypes", "PostsWithDeleted"],["PendingFlags", "FlagTypes"]
                ,["PostTags", "Tags", "TagSynonyms"]
                ,["PostHistory", "PostHistoryTypes"]
                ,["ReviewTasks", "ReviewTaskTypes", "ReviewTaskStates", "ReviewTaskResults", "ReviewRejectionReasons", "ReviewTaskResultTypes"]]
    #lowercase all entries
    ground_truth_clusters = [[x.lower() for x in cluster] for cluster in ground_truth_clusters]
    created_clusters = [['votes', 'users', 'tagsynonyms', 'posttags', 'tags', 'reviewtasks', 'suggestededits', 'posts', 'postswithdeleted', 'postlinks', 'postfeedback', 'suggestededitvotes', 'pendingflags', 'flagtypes', 'posttypes', 'closereasontypes', 'reviewrejectionreasons', 'closeasofftopicreasontypes', 'votetypes', 'reviewtaskresults', 'reviewtaskresulttypes', 'postnotices', 'comments', 'posthistory', 'posthistorytypes', 'reviewtasktypes', 'reviewtaskstates', 'badges', 'postnoticetypes']]
    # -------------------------------------------------------------------
    # (A) We need a mapping of "table -> ground-truth cluster index"
    #     so we can color every table by its ground-truth cluster color.
    # -------------------------------------------------------------------
    def build_table_to_gt_index(gt_clusters):
        """Returns {table_name: cluster_idx} based on ground_truth_clusters list."""
        table2gt = {}
        for c_idx, cluster_list in enumerate(gt_clusters):
            for tbl in cluster_list:
                table2gt[tbl] = c_idx
        return table2gt

    table_to_gt_index = build_table_to_gt_index(ground_truth_clusters)

    # -------------------------------------------------------------------
    # (B) Assign colors based on ground-truth cluster index
    #     (so cluster i => color i from a palette)
    # -------------------------------------------------------------------
    # We can reuse the same palette approach from earlier:
    default_color_list = [
        "red", "blue", "green", "magenta", "cyan",
        "orange", "purple", "brown", "teal", "violet", "olive"
    ]
    # For the ground-truth, if we have k GT clusters:
    k_gt = len(ground_truth_clusters)
    gt_cluster_colors = {}
    for i in range(k_gt):
        gt_cluster_colors[i] = default_color_list[i % len(default_color_list)]

    # -------------------------------------------------------------------
    # (C) Get positions for ground-truth arrangement
    # -------------------------------------------------------------------
    positions_groundtruth = assign_clusters_and_positions_filledcircle(
        ground_truth_clusters,
        big_radius=8.0,
        ring_radius=1.5,
        spacing=2.0,
        cluster_colors=gt_cluster_colors  # color each GT cluster i with gt_cluster_colors[i]
    )

    # -------------------------------------------------------------------
    # (D) Get positions for created arrangement,
    #     BUT color each table by its GT color, not by created cluster index.
    # -------------------------------------------------------------------
    #  1) We do a normal layout for the created clusters (i.e. each created cluster
    #     is placed on a big circle, etc.) BUT we can initially pass "None" for cluster_colors
    #     so that each cluster gets some color, then we overwrite the color with the GT color.
    positions_created = assign_clusters_and_positions_filledcircle(
        created_clusters,
        big_radius=8.0,
        ring_radius=1.5,
        spacing=2.0,
        cluster_colors=None  # We'll override color anyway
    )

    #  2) Overwrite color in positions_created with the ground-truth color
    for tbl, info in positions_created.items():
        # Which GT cluster index does 'tbl' belong to?
        gt_idx = table_to_gt_index[tbl]
        info["color"] = gt_cluster_colors[gt_idx]

    # -------------------------------------------------------------------
    # (E) (OPTIONAL) Normalize each layout so they fit nicely within [-1,1]
    #     Doing them separately allows you to plot them side-by-side
    # -------------------------------------------------------------------
    positions_groundtruth = normalize_coordinates(positions_groundtruth, x_range=(-1, 1), y_range=(-1, 1))
    positions_created     = normalize_coordinates(positions_created,     x_range=(-1, 1), y_range=(-1, 1))

    # -------------------------------------------------------------------
    # (F) Print or otherwise use the results
    # -------------------------------------------------------------------
    print("=== Ground Truth Positions (colors = GT clusters) ===")
    for tbl, info in positions_groundtruth.items():
        print(f"{info},")
    print()

    print("=== Created Clustering Positions (colors = GT clusters) ===")
    for tbl, info in positions_created.items():
        print(f"{info},")


