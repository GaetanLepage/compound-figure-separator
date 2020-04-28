package models;

import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;

/**
 *
 * @author Roger Schaer <roger.schaer at hevs.ch>
 */
public class Figure {

    /* Fields */
    private String filename;
    private Map<Integer, SubFigure> subFigures;

    /* Allowed error margin - currently set at 1/3, meaning there needs to be
    /a 2/3 overlap between a reference figure and a candidate figure */
    public static final double MARGIN = 1.0 / 3;

    /* Constructors */
    public Figure(String filename, Map<Integer, SubFigure> subFigures) {
        this.filename = filename;
        this.subFigures = subFigures;
    }

    public Figure(String filename) {
        this(filename, null);
    }

    /* Calculate score between a reference and a candidate figure */
    public double validateAgainstOtherFigure(Figure other) {

        //Score is number of figures in base data (or more if there are more in the candidate file)
        int maxScore = (other.subFigures.size() > subFigures.size()) ? other.subFigures.size() : subFigures.size();

        //Other figure starts with a score of 0
        int otherScore = 0;

        Map<Integer, Integer> mapFromBaseToOther = new HashMap<Integer, Integer>();

        //Go through all the subfigures and find best correspondence
        for (Entry<Integer, SubFigure> subEntry : subFigures.entrySet()) {
            Integer correspondingFigure = getBestSubFigureMatch(subEntry.getValue(), other.subFigures);

            //If a match was found, put it in the map (if it hasn't been used yet)
            if (correspondingFigure != -1) {
                if (!mapFromBaseToOther.containsValue(correspondingFigure)) {
                    mapFromBaseToOther.put(subEntry.getKey(), correspondingFigure);
                }
            }
        }

        //Score is based on how many figures were correctly identified
        otherScore = mapFromBaseToOther.size();

        return (double) otherScore / maxScore;
    }

    /* Find best matching subfigure in a map of candidates */
    public Integer getBestSubFigureMatch(SubFigure reference, Map<Integer, SubFigure> candidates) {

        Integer maxOverlapFigure = -1;
        double maxOverlap = -1;

        for (Entry<Integer, SubFigure> entry : candidates.entrySet()) {
            double overlap = reference.getOverlapWithOtherFigure(entry.getValue());

            if (overlap >= (1 - 0 - MARGIN) && overlap > maxOverlap) {
                maxOverlap = overlap;
                maxOverlapFigure = entry.getKey();
            }
        }

        return maxOverlapFigure;
    }

    /* Getters & Setters */
    public String getFilename() {
        return filename;
    }

    public void setFilename(String filename) {
        this.filename = filename;
    }

    public Map<Integer, SubFigure> getSubFigures() {
        return subFigures;
    }

    public void setSubFigures(Map<Integer, SubFigure> subFigures) {
        this.subFigures = subFigures;
    }

}
