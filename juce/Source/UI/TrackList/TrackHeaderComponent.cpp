/*
  ==============================================================================

    TrackHeaderComponent.cpp
    
    Implementation of professional DAW-style track headers.

  ==============================================================================
*/

#include "TrackHeaderComponent.h"
#include "../Theme/ThemeManager.h"
#include "../Theme/LayoutConstants.h"

namespace UI
{

namespace
{
constexpr int kDefaultInstrumentComboId = 1;
constexpr int kCurrentInstrumentComboId = 2;
constexpr int kLoadSf2ComboId = 1000000;
constexpr int kLoadSfzComboId = 1000001;
constexpr int kBrowseInstrumentsComboId = 1000002;
constexpr int kGeneratedMidiDisplayComboId = 1000003;
constexpr int kSearchMenuResultId = 1000004;
constexpr int kInstrumentMenuItemBaseId = 200000;
constexpr int kMaxSearchResults = 120;

InstrumentMenuItem makeDefaultInstrumentMenuItem()
{
    InstrumentMenuItem defaultItem;
    defaultItem.id = "default_sine";
    defaultItem.name = "Default (Sine)";
    defaultItem.category = "Default";
    return defaultItem;
}

bool stripLeadingDisplayToken(juce::String& displayName, const juce::String& token)
{
    auto trimmed = displayName.trim();
    if (!trimmed.startsWithIgnoreCase(token) || trimmed.length() <= token.length())
        return false;

    const auto separator = trimmed[token.length()];
    if (separator != '-' && separator != '_' && separator != ' ')
        return false;

    displayName = trimmed.substring(token.length() + 1).trim();
    return true;
}

juce::String menuCategoryLabel(juce::String category)
{
    auto label = category.fromLastOccurrenceOf("·", false, false).trim();
    if (label.isEmpty())
        label = category.trim();
    return label.toLowerCase();
}

juce::String cleanInstrumentMenuDisplayName(juce::String name, const juce::String& category)
{
    const auto originalName = name;
    auto displayName = name.trim();

    bool strippedSourcePrefix = true;
    while (strippedSourcePrefix && displayName.isNotEmpty())
    {
        strippedSourcePrefix = stripLeadingDisplayToken(displayName, "Inst")
                               || stripLeadingDisplayToken(displayName, "RnB");
    }

    const auto label = menuCategoryLabel(category);
    if (label == "bass")
        stripLeadingDisplayToken(displayName, "Bass");
    else if (label == "keys")
        stripLeadingDisplayToken(displayName, "Keys");
    else if (label == "synth")
        stripLeadingDisplayToken(displayName, "Synth");
    else if (label == "pad")
        stripLeadingDisplayToken(displayName, "Pad");
    else if (label == "guitar")
        stripLeadingDisplayToken(displayName, "Guitar");
    else if (label == "strings")
        stripLeadingDisplayToken(displayName, "Strings");
    else if (label == "brass")
        stripLeadingDisplayToken(displayName, "Brass");
    else if (label == "fx")
        stripLeadingDisplayToken(displayName, "FX");
    else if (label == "drums" || label == "drumkits")
    {
        for (const auto& role : { juce::String("Kick"), juce::String("Snare"), juce::String("Hat"), juce::String("Clap") })
        {
            if (stripLeadingDisplayToken(displayName, role))
                break;
        }
    }

    return displayName.isNotEmpty() ? displayName : originalName;
}

juce::String displayCategoryOrOther(const juce::String& category)
{
    const auto trimmed = category.trim();
    return trimmed.isNotEmpty() ? trimmed : juce::String("Other");
}

bool isGeneratedMidiInstrumentId(const juce::String& instrumentId)
{
    return instrumentId.startsWith("gm_program:");
}

bool isDefaultInstrumentId(const juce::String& instrumentId)
{
    return instrumentId == "default_sine";
}

bool isBrowseCatalogItem(const InstrumentMenuItem& item)
{
    return item.id.isNotEmpty()
           && !isDefaultInstrumentId(item.id)
           && !isGeneratedMidiInstrumentId(item.id);
}

juce::String searchSummaryLabel(const juce::String& query, int shown, int total)
{
    auto label = "Search Results for \"" + query + "\" (" + juce::String(shown);
    if (total > shown)
        label += " of " + juce::String(total);
    label += ")";
    return label;
}
}

//==============================================================================
// TrackHeaderComponent
//==============================================================================

TrackHeaderComponent::TrackHeaderComponent(int index)
    : trackIndex(index)
{
    trackName = "Track " + juce::String(index + 1);
    
    // Name label (editable on double-click, single line) - MPC style compact
    nameLabel.setText(trackName, juce::dontSendNotification);
    nameLabel.setFont(juce::Font(Layout::fontSizeSM).boldened());
    nameLabel.setColour(juce::Label::textColourId, AppColours::textPrimary);
    nameLabel.setColour(juce::Label::backgroundColourId, juce::Colours::transparentBlack);
    nameLabel.setEditable(false, true);  // Double-click to edit
    nameLabel.setMinimumHorizontalScale(1.0f);
    nameLabel.setJustificationType(juce::Justification::centredLeft);
    nameLabel.onTextChange = [this]() { onNameEdited(); };
    addAndMakeVisible(nameLabel);
    
    // Instrument/Kit dropdown (MPC style) 
    instrumentCombo.setTextWhenNothingSelected("Select Instrument...");
    instrumentCombo.addItem("Default (Sine)", kDefaultInstrumentComboId);
    instrumentCombo.setSelectedId(kDefaultInstrumentComboId, juce::dontSendNotification);
    instrumentCombo.setColour(juce::ComboBox::backgroundColourId, AppColours::mpcControlBg);
    instrumentCombo.setColour(juce::ComboBox::textColourId, AppColours::textPrimary);
    instrumentCombo.setColour(juce::ComboBox::outlineColourId, AppColours::mpcControlBorder.withAlpha(0.85f));
    instrumentCombo.setColour(juce::ComboBox::focusedOutlineColourId, AppColours::mpcAccentStrong);
    instrumentCombo.setColour(juce::ComboBox::arrowColourId, AppColours::mpcAccentStrong);
    instrumentCombo.onChange = [this]() { onInstrumentSelected(); };
    addAndMakeVisible(instrumentCombo);
    
    // Piano Roll button - opens this track in Piano Roll view
    expandButton.setButtonText(juce::String(juce::CharPointer_UTF8("\xe2\x96\xb6")));  // Always show ▶ (play/edit icon)
    expandButton.setColour(juce::TextButton::buttonColourId, AppColours::mpcControlRaised);
    expandButton.setColour(juce::TextButton::textColourOffId, AppColours::mpcAccentStrong);
    expandButton.setTooltip("Edit in Piano Roll");
    expandButton.onClick = [this]() {
        // Signal to open Piano Roll for this track (expanded=true means "open piano roll")
        listeners.call(&Listener::trackExpandToggled, this, true);
    };
    addAndMakeVisible(expandButton);
    
    // Arm button (record enable) - hidden by default in compact mode, shown on hover/expand
    armButton.setColour(juce::TextButton::buttonColourId, AppColours::mpcControlRaised);
    armButton.setColour(juce::TextButton::buttonOnColourId, juce::Colours::red);
    armButton.setColour(juce::TextButton::textColourOffId, AppColours::textSecondary);
    armButton.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    armButton.setClickingTogglesState(true);
    armButton.setTooltip("Record Arm");
    armButton.onClick = [this]() {
        armed = armButton.getToggleState();
        listeners.call(&Listener::trackArmToggled, this, armed);
        syncToProjectState();
    };
    armButton.setVisible(false);  // Hidden in compact MPC mode
    addAndMakeVisible(armButton);
    
    // Mute button - MPC style compact (small inline toggle)
    muteButton.setColour(juce::TextButton::buttonColourId, AppColours::mpcControlRaised);
    muteButton.setColour(juce::TextButton::buttonOnColourId, AppColours::mpcAmber);  // Amber when active
    muteButton.setColour(juce::TextButton::textColourOffId, AppColours::textSecondary);
    muteButton.setColour(juce::TextButton::textColourOnId, juce::Colours::white);
    muteButton.setClickingTogglesState(true);
    muteButton.setTooltip("Mute");
    muteButton.onClick = [this]() {
        muted = muteButton.getToggleState();
        listeners.call(&Listener::trackMuteToggled, this, muted);
        syncToProjectState();
    };
    addAndMakeVisible(muteButton);
    
    // Solo button - MPC style compact (small inline toggle)
    soloButton.setColour(juce::TextButton::buttonColourId, AppColours::mpcControlRaised);
    soloButton.setColour(juce::TextButton::buttonOnColourId, AppColours::mpcAccentStrong);
    soloButton.setColour(juce::TextButton::textColourOffId, AppColours::textSecondary);
    soloButton.setColour(juce::TextButton::textColourOnId, juce::Colours::black);
    soloButton.setClickingTogglesState(true);
    soloButton.setTooltip("Solo");
    soloButton.onClick = [this]() {
        soloed = soloButton.getToggleState();
        listeners.call(&Listener::trackSoloToggled, this, soloed);
        syncToProjectState();
    };
    addAndMakeVisible(soloButton);
}

TrackHeaderComponent::~TrackHeaderComponent()
{
    if (boundTrackNode.isValid())
        boundTrackNode.removeListener(this);
}

//==============================================================================
void TrackHeaderComponent::setTrackIndex(int index)
{
    trackIndex = index;
    if (trackName.startsWith("Track "))
        setTrackName("Track " + juce::String(index + 1));
    repaint();
}

void TrackHeaderComponent::setTrackName(const juce::String& name)
{
    trackName = name;
    nameLabel.setText(name, juce::dontSendNotification);
    syncToProjectState();
    repaint();
}

void TrackHeaderComponent::setTrackType(TrackType type)
{
    trackType = type;
    repaint();
}

void TrackHeaderComponent::setTrackColour(juce::Colour colour)
{
    trackColour = colour;
    repaint();
}

void TrackHeaderComponent::setSelected(bool isSelected)
{
    if (selected != isSelected)
    {
        selected = isSelected;
        repaint();
    }
}

void TrackHeaderComponent::setExpanded(bool isExpanded)
{
    // No longer toggles visual state - button always shows ▶
    // This method kept for API compatibility but doesn't change appearance
    expanded = isExpanded;
}

void TrackHeaderComponent::setArmed(bool isArmed)
{
    armed = isArmed;
    armButton.setToggleState(isArmed, juce::dontSendNotification);
    repaint();
}

void TrackHeaderComponent::bindToTrackNode(juce::ValueTree node)
{
    if (boundTrackNode.isValid())
        boundTrackNode.removeListener(this);
    
    boundTrackNode = node;
    
    if (boundTrackNode.isValid())
    {
        boundTrackNode.addListener(this);
        updateFromBoundNode();
    }
}

//==============================================================================
void TrackHeaderComponent::addListener(Listener* listener)
{
    listeners.add(listener);
}

void TrackHeaderComponent::removeListener(Listener* listener)
{
    listeners.remove(listener);
}

//==============================================================================
void TrackHeaderComponent::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    auto rowBounds = bounds.toFloat().reduced(1.0f, 1.0f);
    
    // Background - compact, high-contrast MPC-style control strip
    juce::Colour bgColour = AppColours::surfaceSunken;
    if (selected)
        bgColour = AppColours::rowSelected;
    
    g.setGradientFill(juce::ColourGradient(bgColour.brighter(0.08f), rowBounds.getTopLeft(),
                                           bgColour.darker(0.18f), rowBounds.getBottomLeft(), false));
    g.fillRoundedRectangle(rowBounds, Layout::borderRadiusSM);
    
    // Track number box with track color (MPC style)
    trackNumberBounds = bounds.removeFromLeft(Layout::trackHeaderNumberWidth).reduced(2, 3);
    g.setColour(trackColour.darker(0.12f));
    g.fillRoundedRectangle(trackNumberBounds.toFloat(), Layout::borderRadiusSM);
    g.setColour(trackColour.brighter(0.25f));
    g.drawRoundedRectangle(trackNumberBounds.toFloat().reduced(0.5f), Layout::borderRadiusSM, 1.0f);
    
    // Track number text (white on colored background)
    g.setColour(juce::Colours::white);
    g.setFont(juce::Font(Layout::fontSizeSM).boldened());
    g.drawText(juce::String(trackIndex + 1), trackNumberBounds, juce::Justification::centred);

    // Central accent line and compact control well around per-track controls
    const auto contentBounds = getLocalBounds().withTrimmedLeft(Layout::trackHeaderNumberWidth).reduced(4, 5);
    g.setColour(AppColours::mpcControlBg.withAlpha(0.72f));
    g.fillRoundedRectangle(contentBounds.toFloat(), Layout::borderRadiusSM);
    g.setColour((selected ? AppColours::mpcAccentStrong : AppColours::mpcControlBorder).withAlpha(selected ? 0.9f : 0.55f));
    g.drawRoundedRectangle(contentBounds.toFloat().reduced(0.5f), Layout::borderRadiusSM, 1.0f);

    g.setColour((selected ? AppColours::mpcAccentStrong : trackColour).withAlpha(0.85f));
    g.fillRect(contentBounds.withWidth(3));
    
    // Subtle bottom border like MPC
    g.setColour(AppColours::mpcControlBorder.withAlpha(0.65f));
    g.drawHorizontalLine(getHeight() - 1, 0.0f, (float)getWidth());
    
    // Selection highlight
    if (selected)
    {
        g.setColour(AppColours::mpcAccentStrong.withAlpha(0.72f));
        g.drawRoundedRectangle(rowBounds.reduced(0.5f), Layout::borderRadiusSM, 1.5f);
    }
}

void TrackHeaderComponent::resized()
{
    auto bounds = getLocalBounds();
    int height = bounds.getHeight();
    
    // Skip track number box area (painted)
    bounds.removeFromLeft(Layout::trackHeaderNumberWidth);
    
    // Small padding inside the compact control well
    bounds.reduce(Layout::trackHeaderInnerPadding, 5);
    
    // M/S buttons on the right - MPC style compact bordered toggles
    const int buttonSize = Layout::trackHeaderButtonSize;
    const int buttonPadding = Layout::trackHeaderButtonGap;
    auto buttonArea = bounds.removeFromRight((buttonSize * 2) + buttonPadding + 2);
    int buttonY = (height - buttonSize) / 2;
    
    int x = buttonArea.getX();
    muteButton.setBounds(x, buttonY, buttonSize, buttonSize);
    x += buttonSize + buttonPadding;
    soloButton.setBounds(x, buttonY, buttonSize, buttonSize);
    
    // Arm button (hidden in compact mode)
    armButton.setBounds(0, 0, 0, 0);
    
    // Expand button (small, before name)
    expandButton.setBounds(bounds.removeFromLeft(buttonSize).withHeight(buttonSize).withY(buttonY));
    bounds.removeFromLeft(Layout::trackHeaderButtonGap);
    
    // Split remaining space: compact track name plus wider instrument dropdown.
    int nameWidth = juce::jlimit(42, juce::jmax(42, bounds.getWidth() - 52), (int)(bounds.getWidth() * 0.42f));
    int comboWidth = juce::jmax(0, bounds.getWidth() - nameWidth - Layout::trackHeaderButtonGap);
    
    nameLabel.setBounds(bounds.removeFromLeft(nameWidth).withHeight(Layout::trackHeaderCompactRowHeight).withY((height - Layout::trackHeaderCompactRowHeight) / 2));
    bounds.removeFromLeft(Layout::trackHeaderButtonGap);  // gap
    instrumentCombo.setBounds(bounds.removeFromLeft(comboWidth).withHeight(Layout::trackHeaderCompactRowHeight).withY((height - Layout::trackHeaderCompactRowHeight) / 2));
}

void TrackHeaderComponent::mouseDown(const juce::MouseEvent& event)
{
    if (event.mods.isPopupMenu())
    {
        // Show context menu with delete option
        juce::PopupMenu menu;
        menu.addItem(1, "Rename Track");
        menu.addSeparator();
        menu.addItem(2, "Delete Track");
        
        menu.showMenuAsync(juce::PopupMenu::Options().withTargetScreenArea(
            juce::Rectangle<int>(event.getScreenX(), event.getScreenY(), 1, 1)),
            [this](int result) {
                if (result == 1)
                    nameLabel.showEditor();
                else if (result == 2)
                    listeners.call(&Listener::trackDeleteRequested, this);
            });
    }
    else
    {
        listeners.call(&Listener::trackSelected, this);
    }
}

void TrackHeaderComponent::mouseDoubleClick(const juce::MouseEvent& event)
{
    // Double-click on name area to edit
    if (nameLabel.getBounds().contains(event.getPosition()))
    {
        nameLabel.showEditor();
    }
}

//==============================================================================
void TrackHeaderComponent::valueTreePropertyChanged(juce::ValueTree& tree, const juce::Identifier& property)
{
    if (tree == boundTrackNode)
    {
        updateFromBoundNode();
    }
}

void TrackHeaderComponent::updateFromBoundNode()
{
    if (!boundTrackNode.isValid())
        return;
    
    // Update name
    juce::String name = boundTrackNode.getProperty(Project::IDs::name);
    if (name.isNotEmpty())
    {
        trackName = name;
        nameLabel.setText(name, juce::dontSendNotification);
    }
    
    // Update mute/solo
    muted = boundTrackNode.getProperty(Project::IDs::mute);
    soloed = boundTrackNode.getProperty(Project::IDs::solo);
    
    muteButton.setToggleState(muted, juce::dontSendNotification);
    soloButton.setToggleState(soloed, juce::dontSendNotification);

    // Update instrument selection
    auto instIdVar = boundTrackNode.getProperty(Project::IDs::instrumentId);
    juce::String instId = instIdVar.toString();
    if (instId.isEmpty())
        instId = "default_sine";
    if (selectedInstrumentId != instId)
    {
        selectedInstrumentId = instId;
        // If instruments list is already populated, reflect it in the combo.
        // If not populated yet, rebuildInstrumentCombo() will later pick it up.
        if (!instrumentItems.empty())
            setSelectedInstrument(selectedInstrumentId);
    }
    
    repaint();
}

void TrackHeaderComponent::syncToProjectState()
{
    if (!boundTrackNode.isValid())
        return;
    
    boundTrackNode.setProperty(Project::IDs::name, trackName, nullptr);
    boundTrackNode.setProperty(Project::IDs::mute, muted, nullptr);
    boundTrackNode.setProperty(Project::IDs::solo, soloed, nullptr);
    if (!selectedInstrumentId.isEmpty())
        boundTrackNode.setProperty(Project::IDs::instrumentId, selectedInstrumentId, nullptr);
}

void TrackHeaderComponent::onNameEdited()
{
    trackName = nameLabel.getText();
    listeners.call(&Listener::trackNameChanged, this, trackName);
    syncToProjectState();
}

void TrackHeaderComponent::setAvailableInstruments(const std::map<juce::String, std::vector<const mmg::InstrumentDefinition*>>& byCategory)
{
    expansionInstrumentItems.clear();
    
    // Add instruments from each category
    for (const auto& [category, instruments] : byCategory)
    {
        for (const auto* inst : instruments)
        {
            InstrumentMenuItem item;
            item.id = inst->id;
            item.name = cleanInstrumentMenuDisplayName(inst->name, category);
            item.category = category;
            expansionInstrumentItems.push_back(item);
        }
    }
    
    rebuildInstrumentCombo();
}

void TrackHeaderComponent::setAdditionalInstrumentMenuItems(const std::vector<InstrumentMenuItem>& items)
{
    additionalInstrumentItems = items;
    rebuildInstrumentCombo();
}

void TrackHeaderComponent::rebuildInstrumentCombo()
{
    instrumentCombo.clear();
    instrumentItems.clear();
    compactComboInstrumentIndexById.clear();

    const bool hasGeneratedMidiProgram = isGeneratedMidiInstrumentId(selectedInstrumentId);
    const auto generatedMidiInstrumentName = boundTrackNode.isValid()
        ? boundTrackNode.getProperty(Project::IDs::midiInstrumentName).toString()
        : juce::String();
    juce::StringArray seenIds;

    auto appendItemIfUnique = [this, &seenIds](const InstrumentMenuItem& item)
    {
        if (item.id.isEmpty() || seenIds.contains(item.id))
            return;

        instrumentItems.push_back(item);
        seenIds.add(item.id);
    };

    appendItemIfUnique(makeDefaultInstrumentMenuItem());

    for (const auto& item : expansionInstrumentItems)
        appendItemIfUnique(item);

    for (const auto& item : additionalInstrumentItems)
        appendItemIfUnique(item);

    if (hasGeneratedMidiProgram)
    {
        InstrumentMenuItem generatedItem;
        generatedItem.id = selectedInstrumentId;
        generatedItem.name = generatedMidiInstrumentName.isNotEmpty()
            ? ("MIDI: " + generatedMidiInstrumentName)
            : ("MIDI: " + selectedInstrumentId.fromFirstOccurrenceOf(":", false, false));
        generatedItem.category = "Generated MIDI";
        if (!seenIds.contains(generatedItem.id))
        {
            instrumentItems.push_back(generatedItem);
            seenIds.add(generatedItem.id);
        }
    }

    // Compact root list: current/default selection plus actions only. The full
    // catalog stays in instrumentItems and is exposed through the browser menu.
    instrumentCombo.clear();
    instrumentCombo.addSectionHeading("Track Instrument");

    const auto defaultIndex = findInstrumentItemIndex("default_sine");
    if (defaultIndex >= 0)
    {
        instrumentCombo.addItem("Default (Sine)", kDefaultInstrumentComboId);
        compactComboInstrumentIndexById[kDefaultInstrumentComboId] = defaultIndex;
    }

    const auto currentIndex = findInstrumentItemIndex(selectedInstrumentId);
    if (currentIndex >= 0 && !isDefaultInstrumentId(selectedInstrumentId) && !hasGeneratedMidiProgram)
    {
        instrumentCombo.addSeparator();
        instrumentCombo.addSectionHeading("Current Instrument");
        instrumentCombo.addItem(currentInstrumentDisplayName(), kCurrentInstrumentComboId);
        compactComboInstrumentIndexById[kCurrentInstrumentComboId] = currentIndex;
    }
    else if (hasGeneratedMidiProgram)
    {
        instrumentCombo.addSeparator();
        instrumentCombo.addSectionHeading("Generated MIDI");
        instrumentCombo.addItem(currentInstrumentDisplayName(), kGeneratedMidiDisplayComboId);
    }

    instrumentCombo.addSeparator();
    instrumentCombo.addSectionHeading("Instrument Catalog");
    instrumentCombo.addItem("Browse/Search Instruments...", kBrowseInstrumentsComboId);

    // Add SoundFont loading options
    instrumentCombo.addSeparator();
    instrumentCombo.addSectionHeading("External Instruments");
    instrumentCombo.addItem("Load SF2/SoundFont...", kLoadSf2ComboId);  // Special ID for SF2
    instrumentCombo.addItem("Load SFZ Instrument...", kLoadSfzComboId); // Special ID for SFZ
    
    // Select the current instrument or default
    if (!selectedInstrumentId.isEmpty())
        setSelectedInstrument(selectedInstrumentId);
    else if (!instrumentItems.empty())
        instrumentCombo.setSelectedId(1, juce::dontSendNotification);
}

void TrackHeaderComponent::onInstrumentSelected()
{
    int selectedId = instrumentCombo.getSelectedId();
    if (selectedId == 0)
        return;

    if (auto mappedIndex = compactComboInstrumentIndexById.find(selectedId); mappedIndex != compactComboInstrumentIndexById.end())
    {
        selectInstrumentMenuItem(mappedIndex->second);
        return;
    }

    if (selectedId == kBrowseInstrumentsComboId)
    {
        if (!selectedInstrumentId.isEmpty())
            setSelectedInstrument(selectedInstrumentId);
        showInstrumentBrowserMenu();
        return;
    }

    if (selectedId == kGeneratedMidiDisplayComboId)
    {
        // Generated-MIDI pseudo-instruments are display-only. Keep the current
        // selection visible without trying to load it as a catalog instrument.
        setSelectedInstrument(selectedInstrumentId);
        return;
    }
    
    // Handle special SoundFont loading options
    if (selectedId == kLoadSf2ComboId)
    {
        listeners.call(&Listener::trackLoadSF2Requested, this);
        // Reset combo to previous selection
        if (!selectedInstrumentId.isEmpty())
            setSelectedInstrument(selectedInstrumentId);
        return;
    }
    else if (selectedId == kLoadSfzComboId)
    {
        listeners.call(&Listener::trackLoadSFZRequested, this);
        // Reset combo to previous selection
        if (!selectedInstrumentId.isEmpty())
            setSelectedInstrument(selectedInstrumentId);
        return;
    }
}

void TrackHeaderComponent::setSelectedInstrument(const juce::String& instrumentId)
{
    selectedInstrumentId = instrumentId;

    if (isDefaultInstrumentId(instrumentId) && compactComboInstrumentIndexById.count(kDefaultInstrumentComboId) > 0)
    {
        instrumentCombo.setSelectedId(kDefaultInstrumentComboId, juce::dontSendNotification);
        return;
    }

    if (isGeneratedMidiInstrumentId(instrumentId))
    {
        instrumentCombo.setSelectedId(kGeneratedMidiDisplayComboId, juce::dontSendNotification);
        return;
    }

    if (compactComboInstrumentIndexById.count(kCurrentInstrumentComboId) > 0)
    {
        const auto currentIndex = compactComboInstrumentIndexById[kCurrentInstrumentComboId];
        if (currentIndex >= 0 && currentIndex < (int)instrumentItems.size()
            && instrumentItems[(size_t)currentIndex].id == instrumentId)
        {
            instrumentCombo.setSelectedId(kCurrentInstrumentComboId, juce::dontSendNotification);
            return;
        }
    }
    
    // If not found, select default
    if (compactComboInstrumentIndexById.count(kDefaultInstrumentComboId) > 0)
        instrumentCombo.setSelectedId(kDefaultInstrumentComboId, juce::dontSendNotification);
}

void TrackHeaderComponent::showInstrumentBrowserMenu()
{
    juce::PopupMenu menu;
    menu.addItem(kSearchMenuResultId, "Search Instruments...");
    menu.addSeparator();

    std::map<juce::String, std::vector<int>> itemsByCategory;
    juce::StringArray seenIds;

    for (int i = 0; i < (int)instrumentItems.size(); ++i)
    {
        const auto& item = instrumentItems[(size_t)i];
        if (!isBrowseCatalogItem(item) || seenIds.contains(item.id))
            continue;

        itemsByCategory[displayCategoryOrOther(item.category)].push_back(i);
        seenIds.add(item.id);
    }

    if (itemsByCategory.empty())
    {
        menu.addItem(kSearchMenuResultId + 1, "No catalog instruments available", false, false);
    }
    else
    {
        for (const auto& [category, itemIndexes] : itemsByCategory)
        {
            juce::PopupMenu categoryMenu;
            for (const auto itemIndex : itemIndexes)
            {
                const auto& item = instrumentItems[(size_t)itemIndex];
                categoryMenu.addItem(kInstrumentMenuItemBaseId + itemIndex,
                                     item.name,
                                     true,
                                     item.id == selectedInstrumentId);
            }

            menu.addSubMenu(category, categoryMenu, true);
        }
    }

    juce::Component::SafePointer<TrackHeaderComponent> safeThis(this);
    menu.showMenuAsync(juce::PopupMenu::Options()
        .withTargetComponent(&instrumentCombo)
        .withMinimumWidth(240),
        [safeThis](int result) mutable {
            if (safeThis == nullptr || result == 0)
                return;

            if (result == kSearchMenuResultId)
            {
                safeThis->showInstrumentSearchDialog();
                return;
            }

            if (result >= kInstrumentMenuItemBaseId)
                safeThis->selectInstrumentMenuItem(result - kInstrumentMenuItemBaseId);
        });
}

void TrackHeaderComponent::showInstrumentSearchDialog()
{
    auto* dialog = new juce::AlertWindow("Search Instruments",
        "Enter part of an instrument name or category. Multiple words narrow the search.",
        juce::MessageBoxIconType::QuestionIcon);

    dialog->addTextEditor("query", "", "Search:");
    dialog->addButton("Search", 1, juce::KeyPress(juce::KeyPress::returnKey));
    dialog->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));

    juce::Component::SafePointer<TrackHeaderComponent> safeThis(this);
    dialog->enterModalState(true, juce::ModalCallbackFunction::create(
        [safeThis, dialog](int result) mutable {
            if (result == 1 && safeThis != nullptr)
            {
                auto query = dialog->getTextEditorContents("query").trim();
                if (query.isNotEmpty())
                    safeThis->showInstrumentSearchResults(query);
                else
                    safeThis->showInstrumentBrowserMenu();
            }

            delete dialog;
        }), true);
}

void TrackHeaderComponent::showInstrumentSearchResults(const juce::String& query)
{
    juce::StringArray tokens;
    tokens.addTokens(query.toLowerCase(), " \t\r\n", "\"'");
    tokens.removeEmptyStrings();

    std::map<juce::String, std::vector<int>> matchesByCategory;
    int totalMatches = 0;

    for (int i = 0; i < (int)instrumentItems.size(); ++i)
    {
        const auto& item = instrumentItems[(size_t)i];
        if (!isBrowseCatalogItem(item))
            continue;

        const auto haystack = (item.name + " " + item.category).toLowerCase();
        bool matches = true;
        for (const auto& token : tokens)
        {
            if (!haystack.contains(token))
            {
                matches = false;
                break;
            }
        }

        if (!matches)
            continue;

        ++totalMatches;
        if (totalMatches <= kMaxSearchResults)
            matchesByCategory[displayCategoryOrOther(item.category)].push_back(i);
    }

    juce::PopupMenu menu;
    menu.addItem(kSearchMenuResultId, "Search Again...");
    menu.addSeparator();

    const auto shownMatches = juce::jmin(totalMatches, kMaxSearchResults);
    menu.addItem(kSearchMenuResultId + 3, searchSummaryLabel(query, shownMatches, totalMatches), false, false);

    if (totalMatches == 0)
    {
        menu.addItem(kSearchMenuResultId + 1, "No matching instruments", false, false);
    }
    else
    {
        for (const auto& [category, itemIndexes] : matchesByCategory)
        {
            juce::PopupMenu categoryMenu;
            for (const auto itemIndex : itemIndexes)
            {
                const auto& item = instrumentItems[(size_t)itemIndex];
                categoryMenu.addItem(kInstrumentMenuItemBaseId + itemIndex,
                                     item.name,
                                     true,
                                     item.id == selectedInstrumentId);
            }

            menu.addSubMenu(category, categoryMenu, true);
        }

        if (totalMatches > kMaxSearchResults)
        {
            menu.addSeparator();
            menu.addItem(kSearchMenuResultId + 2,
                         "Refine search to see more than " + juce::String(kMaxSearchResults) + " matches",
                         false,
                         false);
        }
    }

    juce::Component::SafePointer<TrackHeaderComponent> safeThis(this);
    menu.showMenuAsync(juce::PopupMenu::Options()
        .withTargetComponent(&instrumentCombo)
        .withMinimumWidth(260),
        [safeThis](int result) mutable {
            if (safeThis == nullptr || result == 0)
                return;

            if (result == kSearchMenuResultId)
            {
                safeThis->showInstrumentSearchDialog();
                return;
            }

            if (result >= kInstrumentMenuItemBaseId)
                safeThis->selectInstrumentMenuItem(result - kInstrumentMenuItemBaseId);
        });
}

void TrackHeaderComponent::selectInstrumentMenuItem(int instrumentItemIndex)
{
    if (instrumentItemIndex < 0 || instrumentItemIndex >= (int)instrumentItems.size())
        return;

    const auto& item = instrumentItems[(size_t)instrumentItemIndex];
    if (item.id.isEmpty() || isGeneratedMidiInstrumentId(item.id))
        return;

    selectedInstrumentId = item.id;
    rebuildInstrumentCombo();
    setSelectedInstrument(selectedInstrumentId);
    listeners.call(&Listener::trackInstrumentChanged, this, selectedInstrumentId);
    syncToProjectState();
}

int TrackHeaderComponent::findInstrumentItemIndex(const juce::String& instrumentId) const
{
    if (instrumentId.isEmpty())
        return -1;

    for (size_t i = 0; i < instrumentItems.size(); ++i)
    {
        if (instrumentItems[i].id == instrumentId)
            return (int)i;
    }

    return -1;
}

juce::String TrackHeaderComponent::currentInstrumentDisplayName() const
{
    if (isGeneratedMidiInstrumentId(selectedInstrumentId))
    {
        const auto generatedMidiInstrumentName = boundTrackNode.isValid()
            ? boundTrackNode.getProperty(Project::IDs::midiInstrumentName).toString()
            : juce::String();
        return generatedMidiInstrumentName.isNotEmpty()
            ? ("MIDI: " + generatedMidiInstrumentName)
            : ("MIDI: " + selectedInstrumentId.fromFirstOccurrenceOf(":", false, false));
    }

    const auto index = findInstrumentItemIndex(selectedInstrumentId);
    if (index >= 0)
        return instrumentItems[(size_t)index].name;

    if (isDefaultInstrumentId(selectedInstrumentId) || selectedInstrumentId.isEmpty())
        return "Default (Sine)";

    if (selectedInstrumentId.startsWith("sample:"))
    {
        const auto samplePath = selectedInstrumentId.fromFirstOccurrenceOf(":", false, false);
        auto sampleName = juce::File(samplePath).getFileNameWithoutExtension();
        if (sampleName.isNotEmpty())
            return sampleName;
    }

    return selectedInstrumentId;
}

//==============================================================================
// TrackListComponent
//==============================================================================

TrackListComponent::TrackListComponent()
{
    viewport.setViewedComponent(&contentComponent, false);
    viewport.setScrollBarsShown(true, false);
    addAndMakeVisible(viewport);
    
    // MPC-style section headers
    midiSectionHeader = std::make_unique<TrackSectionHeader>("MIDI", juce::Colour(0xFF00D4AA));
    contentComponent.addAndMakeVisible(*midiSectionHeader);
    
    audioSectionHeader = std::make_unique<TrackSectionHeader>("AUDIO", juce::Colour(0xFF2196F3));
    contentComponent.addAndMakeVisible(*audioSectionHeader);
    audioSectionHeader->setVisible(false);  // Hidden until we have audio tracks
    
    // Add track button - MPC style
    addTrackButton.setColour(juce::TextButton::buttonColourId, ThemeManager::getSurface());
    addTrackButton.setColour(juce::TextButton::textColourOffId, ThemeManager::getCurrentScheme().textSecondary);
    addTrackButton.onClick = [this]() {
        addTrack(TrackType::MIDI);
    };
    addAndMakeVisible(addTrackButton);
    
    // Create default tracks
    setTrackCount(4);
}

TrackListComponent::~TrackListComponent()
{
    for (auto* header : trackHeaders)
        header->removeListener(this);
}

//==============================================================================
void TrackListComponent::setTrackCount(int count)
{
    // Remove excess
    while (trackHeaders.size() > count)
    {
        trackHeaders.getLast()->removeListener(this);
        trackHeaders.removeLast();
    }
    
    // Add new tracks
    while (trackHeaders.size() < count)
    {
        int index = trackHeaders.size();
        auto* header = trackHeaders.add(new TrackHeaderComponent(index));
        header->setTrackColour(getNextTrackColour());
        header->addListener(this);
        contentComponent.addAndMakeVisible(header);
        
        // Set available instruments if we have them
        if (!availableInstruments.empty())
            header->setAvailableInstruments(availableInstruments);
        if (!additionalInstrumentMenuItems.empty())
            header->setAdditionalInstrumentMenuItems(additionalInstrumentMenuItems);
    }
    
    updateLayout();
    listeners.call(&Listener::trackCountChanged, count);
}

void TrackListComponent::addTrack(TrackType type, const juce::String& name)
{
    int index = trackHeaders.size();
    auto* header = trackHeaders.add(new TrackHeaderComponent(index));
    header->setTrackType(type);
    header->setTrackColour(getNextTrackColour());
    
    if (name.isNotEmpty())
        header->setTrackName(name);
    else
        header->setTrackName("Track " + juce::String(index + 1));
    
    header->addListener(this);
    contentComponent.addAndMakeVisible(header);
    
    // Set available instruments if we have them
    if (!availableInstruments.empty())
        header->setAvailableInstruments(availableInstruments);
    if (!additionalInstrumentMenuItems.empty())
        header->setAdditionalInstrumentMenuItems(additionalInstrumentMenuItems);
    
    // Also add to project state if bound
    if (projectState)
    {
        // getTrackNode will create the track if it doesn't exist
        auto trackNode = projectState->getTrackNode(index);
        if (trackNode.isValid())
        {
            // Set the track name in the project state
            trackNode.setProperty(Project::IDs::name, header->getTrackName(), nullptr);
            header->bindToTrackNode(trackNode);
        }
    }
    
    updateLayout();
    listeners.call(&Listener::trackCountChanged, trackHeaders.size());
}

void TrackListComponent::removeTrack(int index)
{
    if (index >= 0 && index < trackHeaders.size())
    {
        trackHeaders[index]->removeListener(this);
        trackHeaders.remove(index);
        
        // Re-index remaining tracks
        for (int i = index; i < trackHeaders.size(); ++i)
            trackHeaders[i]->setTrackIndex(i);
        
        // Adjust selection
        if (selectedTrackIndex >= trackHeaders.size())
            selectedTrackIndex = trackHeaders.size() - 1;
        
        updateLayout();
        listeners.call(&Listener::trackCountChanged, trackHeaders.size());
    }
}

TrackHeaderComponent* TrackListComponent::getTrackHeader(int index)
{
    if (index >= 0 && index < trackHeaders.size())
        return trackHeaders[index];
    return nullptr;
}

void TrackListComponent::selectTrack(int index)
{
    if (index >= 0 && index < trackHeaders.size())
    {
        // Deselect previous
        if (selectedTrackIndex >= 0 && selectedTrackIndex < trackHeaders.size())
            trackHeaders[selectedTrackIndex]->setSelected(false);
        
        selectedTrackIndex = index;
        trackHeaders[index]->setSelected(true);
        
        listeners.call(&Listener::trackSelectionChanged, index);
    }
}

void TrackListComponent::bindToProject(Project::ProjectState& state)
{
    projectState = &state;
    
    // Get tracks from project state
    auto mixerNode = state.getMixerNode();
    juce::StringArray trackNames;
    
    for (const auto& child : mixerNode)
    {
        if (child.hasType(Project::IDs::TRACK))
            trackNames.add(child.getProperty(Project::IDs::name));
    }
    
    // If no tracks, create defaults
    if (trackNames.isEmpty())
    {
        for (int i = 0; i < 4; ++i)
            trackNames.add("Track " + juce::String(i + 1));
    }
    
    // Update track count and bind to nodes
    setTrackCount(trackNames.size());
    
    for (int i = 0; i < trackHeaders.size(); ++i)
    {
        trackHeaders[i]->setTrackName(trackNames[i]);
        
        auto trackNode = state.getTrackNode(i);
        if (trackNode.isValid())
            trackHeaders[i]->bindToTrackNode(trackNode);
    }
    
    // Select first track
    if (!trackHeaders.isEmpty())
        selectTrack(0);
}

//==============================================================================
void TrackListComponent::addListener(Listener* listener)
{
    listeners.add(listener);
}

void TrackListComponent::removeListener(Listener* listener)
{
    listeners.remove(listener);
}

//==============================================================================
void TrackListComponent::paint(juce::Graphics& g)
{
    g.fillAll(ThemeManager::getCurrentScheme().background);
}

void TrackListComponent::resized()
{
    auto bounds = getLocalBounds();
    
    // Add track button at bottom
    addTrackButton.setBounds(bounds.removeFromBottom(24).reduced(4, 2));
    
    // Viewport takes rest
    viewport.setBounds(bounds);
    
    updateLayout();
}

//==============================================================================
void TrackListComponent::trackSelected(TrackHeaderComponent* track)
{
    selectTrack(track->getTrackIndex());
}

void TrackListComponent::trackExpandToggled(TrackHeaderComponent* track, bool expanded)
{
    updateLayout();
    listeners.call(&Listener::trackExpandedChanged, track->getTrackIndex(), expanded);
}

void TrackListComponent::trackArmToggled(TrackHeaderComponent* track, bool armed)
{
    DBG("Track " + juce::String(track->getTrackIndex() + 1) + " arm: " + (armed ? "ON" : "OFF"));
}

void TrackListComponent::trackMuteToggled(TrackHeaderComponent* track, bool muted)
{
    DBG("Track " + juce::String(track->getTrackIndex() + 1) + " mute: " + (muted ? "ON" : "OFF"));
}

void TrackListComponent::trackSoloToggled(TrackHeaderComponent* track, bool soloed)
{
    DBG("Track " + juce::String(track->getTrackIndex() + 1) + " solo: " + (soloed ? "ON" : "OFF"));
}

void TrackListComponent::trackNameChanged(TrackHeaderComponent* track, const juce::String& newName)
{
    DBG("Track " + juce::String(track->getTrackIndex() + 1) + " renamed to: " + newName);
}

void TrackListComponent::trackDeleteRequested(TrackHeaderComponent* track)
{
    int index = track->getTrackIndex();
    
    // Don't allow deleting the last track
    if (trackHeaders.size() <= 1)
    {
        DBG("Cannot delete the last track");
        return;
    }
    
    DBG("Track " + juce::String(index + 1) + " delete requested");
    removeTrack(index);
}

void TrackListComponent::trackInstrumentChanged(TrackHeaderComponent* track, const juce::String& instrumentId)
{
    int index = track->getTrackIndex();
    DBG("Track " + juce::String(index + 1) + " instrument changed to: " + instrumentId);
    listeners.call(&Listener::trackInstrumentSelected, index, instrumentId);
}

void TrackListComponent::trackLoadSF2Requested(TrackHeaderComponent* track)
{
    int index = track->getTrackIndex();
    DBG("Track " + juce::String(index + 1) + " load SF2 requested");
    listeners.call(&Listener::trackLoadSF2Requested, index);
}

void TrackListComponent::trackLoadSFZRequested(TrackHeaderComponent* track)
{
    int index = track->getTrackIndex();
    DBG("Track " + juce::String(index + 1) + " load SFZ requested");
    listeners.call(&Listener::trackLoadSFZRequested, index);
}

void TrackListComponent::setAvailableInstruments(const std::map<juce::String, std::vector<const mmg::InstrumentDefinition*>>& byCategory)
{
    availableInstruments = byCategory;
    
    // Propagate to all track headers
    for (auto* header : trackHeaders)
    {
        header->setAvailableInstruments(byCategory);
    }
}

void TrackListComponent::setAdditionalInstrumentMenuItems(const std::vector<InstrumentMenuItem>& items)
{
    additionalInstrumentMenuItems = items;

    // Propagate to all track headers
    for (auto* header : trackHeaders)
    {
        header->setAdditionalInstrumentMenuItems(items);
    }
}

//==============================================================================
void TrackListComponent::updateLayout()
{
    int y = 0;
    int width = viewport.getWidth() - viewport.getScrollBarThickness();
    if (width <= 0) width = 200;  // Fallback width
    
    // Count MIDI and Audio tracks
    int midiCount = 0;
    int audioCount = 0;
    for (auto* header : trackHeaders)
    {
        if (header->getTrackType() == TrackType::Audio)
            audioCount++;
        else
            midiCount++;
    }
    
    // MIDI section header (always visible if we have MIDI tracks)
    if (midiCount > 0)
    {
        midiSectionHeader->setVisible(true);
        midiSectionHeader->setBounds(0, y, width, sectionHeaderHeight);
        y += sectionHeaderHeight;
        
        // Layout MIDI tracks - all uniform height
        for (auto* header : trackHeaders)
        {
            if (header->getTrackType() != TrackType::Audio)
            {
                header->setBounds(0, y, width, trackHeight);
                y += trackHeight;
            }
        }
    }
    
    // AUDIO section header (visible if we have audio tracks)
    if (audioCount > 0)
    {
        audioSectionHeader->setVisible(true);
        audioSectionHeader->setBounds(0, y, width, sectionHeaderHeight);
        y += sectionHeaderHeight;
        
        // Layout Audio tracks - all uniform height
        for (auto* header : trackHeaders)
        {
            if (header->getTrackType() == TrackType::Audio)
            {
                header->setBounds(0, y, width, trackHeight);
                y += trackHeight;
            }
        }
    }
    else
    {
        audioSectionHeader->setVisible(false);
    }
    
    contentComponent.setSize(width, y);
}

juce::Colour TrackListComponent::getNextTrackColour()
{
    int index = trackHeaders.size() % trackColours.size();
    return trackColours[index];
}

} // namespace UI
