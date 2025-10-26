import { app } from "/scripts/app.js";

const LOCALSTORAGE_PREFIX = "wanPromptProfile_";

// ====================================================================================================
// Helper Functions
// ====================================================================================================
function findWidgetByName(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

let statusClearTimeout = null;
function showStatusMessage(node, message, isError = false) {
    if (isError) {
        console.error(`[WanVideoLooperPrompts] ${message}`);
        alert(`Error: ${message}`);
    } else {
        console.log(`[WanVideoLooperPrompts] ${message}`);
    }
}

// ====================================================================================================
// Get/Set Data Logic
// ====================================================================================================

const WIDGET_NAMES_TO_SAVE = ["prompts", "negative_prompt", "enable_prefix", "enable_suffix", "prefix", "suffix"];

function getWidgetData(node) {
    const data = {};
    WIDGET_NAMES_TO_SAVE.forEach(name => {
        const widget = findWidgetByName(node, name);
        if (widget) {
            data[name] = widget.value;
        } else {
            console.warn(`[WanVideoLooperPrompts] Widget not found during save: ${name}`);
        }
    });
    return data;
}

function setWidgetData(node, data) {
    if (!data) return;
    WIDGET_NAMES_TO_SAVE.forEach(name => {
        const widget = findWidgetByName(node, name);
        if (widget && data[name] !== undefined) {
             if (widget.type === "combo" || widget.type === "toggle") {
                if (widget.options?.values?.includes(data[name])) {
                     widget.value = data[name];
                }
            } else {
                widget.value = data[name];
            }
        } else {
            console.warn(`[WanVideoLooperPrompts] Widget not found during load: ${name}`);
        }
    });
    
    node.setDirtyCanvas(true, true);
    if (node.flags.collapsed) {
        node.setSize(node.computeSize());
    }
    app.graph.setDirtyCanvas(true, true);
}

// ====================================================================================================
// Profile Save/Load/Delete Logic
// ====================================================================================================

function refreshProfileList(node) {
    const profileListWidget = findWidgetByName(node, "profile_list");
    if (!profileListWidget) return;

    const currentSelection = profileListWidget.value;
    const profileNames = [""];
    try {
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key && key.startsWith(LOCALSTORAGE_PREFIX)) {
                profileNames.push(key.substring(LOCALSTORAGE_PREFIX.length));
            }
        }
    } catch (e) {
         console.error("[WanVideoLooperPrompts] Error accessing localStorage:", e);
         showStatusMessage(node, "Error accessing storage.", true);
         profileNames = ["Error reading storage"];
    }

    profileListWidget.options.values = profileNames.sort();
    profileListWidget.value = profileListWidget.options.values.includes(currentSelection) ? currentSelection : "";
    node.setDirtyCanvas(true, true);
}

function saveProfile(node) {
    const nameWidget = findWidgetByName(node, "new_profile_name");
    const profileName = nameWidget?.value;

    if (!profileName || !profileName.trim()) {
        alert("Save cancelled: Profile name cannot be empty.");
        return;
    }
    const cleanName = profileName.trim();
    const key = LOCALSTORAGE_PREFIX + cleanName;
    const data = getWidgetData(node);

    if (localStorage.getItem(key)) {
        if (!confirm(`Profile "${cleanName}" already exists. Overwrite?`)) {
            showStatusMessage(node, "Save cancelled.", false);
            return;
        }
    }
    try {
        const jsonData = JSON.stringify(data);
        localStorage.setItem(key, jsonData);
        alert(`Profile "${cleanName}" saved successfully!`);
        nameWidget.value = "";
        refreshProfileList(node);
        const profileListWidget = findWidgetByName(node, "profile_list");
        if(profileListWidget) profileListWidget.value = cleanName;
    } catch (error) {
        console.error("Error saving profile:", error);
        alert("Error saving profile. Check console for details.");
    }
}

function loadProfile(node) {
    const profileListWidget = findWidgetByName(node, "profile_list");
    const profileName = profileListWidget?.value;

    if (!profileName) {
        alert("No profile selected from the list.");
        return;
    }
    const key = LOCALSTORAGE_PREFIX + profileName;
    try {
        const jsonData = localStorage.getItem(key);
        if (jsonData) {
            const data = JSON.parse(jsonData);
            setWidgetData(node, data);
            console.log(`[WanVideoLooperPrompts] Profile "${profileName}" loaded.`); // No popup
        } else {
            alert(`Error: Profile "${profileName}" not found in storage.`);
            refreshProfileList(node);
        }
    } catch (error) {
        console.error("Error loading profile:", error);
        alert("Error loading profile data. Check console.");
    }
}

function deleteProfile(node) {
    const profileListWidget = findWidgetByName(node, "profile_list");
    const profileName = profileListWidget?.value;

    if (!profileName) {
        alert("No profile selected from the list to delete.");
        return;
    }
    if (confirm(`Are you sure you want to delete "${profileName}"?`)) {
        const key = LOCALSTORAGE_PREFIX + profileName;
        try {
            localStorage.removeItem(key);
            alert(`Profile "${profileName}" deleted successfully!`);
            refreshProfileList(node);
        } catch (error) {
            console.error("Error deleting profile:", error);
            alert("Error deleting profile. Check console.");
        }
    }
}

// ====================================================================================================
// ComfyUI Extension Registration
// ====================================================================================================

app.registerExtension({
    name: "WanVideoLooper.PromptsNodeSimple",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "WanVideoLooperPrompts") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);


                this.addWidget("text", "new_profile_name", "", null, { placeholder: "New profile name..." });
                this.addWidget( "button", "Save Profile As", null, () => saveProfile(this));
                this.addWidget( "button", "Refresh List", null, () => refreshProfileList(this));
                this.addWidget( "button", "Load Selected", null, () => loadProfile(this));
                this.addWidget( "combo", "profile_list", "", () => {}, { values: [""] });
                this.addWidget( "button", "🗑️ Delete Profile", null, () => deleteProfile(this));
                

                setTimeout(() => refreshProfileList(this), 100);
            };
        }
    },
});