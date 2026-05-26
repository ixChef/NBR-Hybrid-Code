from pathlib import Path

projectRoot = Path(__file__).resolve().parents[1]
workspaceRoot = projectRoot.parent

tifuProfile = {
    "profileName": "tifu_profile",
    "repoRoot": workspaceRoot / "time_dependent_nbr",
    "sourceDataRoot": workspaceRoot / "time_dependent_nbr" / "data",
    "outputRoot": projectRoot / "data" / "tifu_profile",
    "minBasketsPerUser": 3,
    "minItemsPerUser": 0,
    "minUsersPerItem": 5,
    "maxUsersNum": 0,
    "maxItemsNum": 0,
    "splitStrategy": "leave_two_baskets",
    "randomBaskets": False,
    "randomState": 42,
    "useSymlink": True,
    "datasets": {
        "dunnhumby": {
            "sourceRawFile": "transaction_data.csv"
        },
        "tafeng": {
            "sourceRawFile": "ta_feng_all_months_merged.csv"
        }
    }
}

taiwProfile = {
    "profileName": "taiw_profile",
    "repoRoot": workspaceRoot / "time_aware_item_weighting",
    "sourceDataRoot": workspaceRoot / "time_aware_item_weighting" / "data",
    "outputRoot": projectRoot / "data" / "taiw_profile",
    "userMin": 5,
    "itemMin": 10,
    "applyNoiseUserFilter": True,
    "useSymlink": True,
    "datasets": {
        "dunnhumby": {
            "sourceRawFile": "dunnhumby.txt"
        },
        "tafeng": {
            "sourceRawFile": "tafeng.txt"
        }
    }
}

tifuHybridProfile = {
    "profileName": "tifu_hybrid_profile",
    "repoRoot": workspaceRoot / "time_aware_item_weighting",
    "sourceDataRoot": projectRoot / "data" / "tifu_profile_taiw_hybrid",
    "outputRoot": projectRoot / "data" / "tifu_profile_taiw_hybrid",
    "useSymlink": False,
    "datasets": {
        "dunnhumby": {
            "sourceRawFile": None
        },
        "tafeng": {
            "sourceRawFile": None
        }
    }
}

def getProfile(profileName: str):
    profiles = {
        "tifu_profile": tifuProfile,
        "taiw_profile": taiwProfile,
        "tifu_hybrid_profile": tifuHybridProfile,
    }
    if profileName not in profiles:
        raise ValueError(f"Unknown profile: {profileName}")
    return profiles[profileName]