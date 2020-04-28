package cfsevaluation;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import models.Figure;
import models.Point;
import models.SubFigure;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;

/**
 *
 * @author Roger Schaer <roger.schaer at hevs.ch>
 */
public class EvaluateCompoundFigureSeparation {

    /* Contants - XML element & attribute names */
    private static final String ANNOTATION = "annotation";
    private static final String FILENAME = "filename";
    private static final String ZONE = "object";
    private static final String POINT = "point";
    private static final String X = "x";
    private static final String Y = "y";

    public static void main(String[] args) throws IOException{

        /* Args explanation */
        if(args.length != 2){
            System.err.println("Args must have a length of exactly 2 !");
            System.err.println("USAGE : java -jar CFSEvaluation.jar [path_of_ground_truth] [path_of_candidate]");
            return;
        }

        /* Read arguments */
        String gtFileName = args[0];
        String cdFileName = args[1];

        File gtFile = new File(gtFileName);
        File cdFile = new File(cdFileName);

        double maxScore = 0.0;
        double candidateScore = 0.0;

        //Parse XML files
        Document gtData = Jsoup.parse(gtFile, Charset.defaultCharset().name());
        Document cdData = Jsoup.parse(cdFile, Charset.defaultCharset().name());

        //Transform into a map of filename->subfigures map
        Map<String, Figure> gtFigures = getFigures(gtData);
        maxScore = gtFigures.size();

        Map<String, Figure> cdFigures = getFigures(cdData);

        //Scores maps
        Map<String, Double> scoresMap = new LinkedHashMap<String, Double>();

        //Compare all figures
        for(Entry<String, Figure> entry : gtFigures.entrySet()){
            String filename = entry.getKey();
            Figure gtFigure = entry.getValue();

            //If the file is present in the candidate, validate it
            if(cdFigures.containsKey(filename)){
                Figure cdFigure = cdFigures.get(filename);

                double figureScore = gtFigure.validateAgainstOtherFigure(cdFigure);

                scoresMap.put(filename, figureScore);

                candidateScore += figureScore;
            }

        }

        /* Output */
        System.out.println("Max score is : " + maxScore);
        System.out.println("Candidate achieved : " + candidateScore);
        System.out.println("Resulting percentage : " + (candidateScore / maxScore) * 100 + "% accuracy");

    }

    //Transform XML doc to a map of figures
    public static Map<String, Figure> getFigures(Document data) {

        //Empty figures map
        Map<String, Figure> figures = new LinkedHashMap<String, Figure>();

        //Get all GT figures
        for (Element annotation : data.select(ANNOTATION)) {
            //Key
            String imageName = annotation.select(FILENAME).first().text();

            //Value
            int i = 0;
            Map<Integer, SubFigure> subFigures = new LinkedHashMap<Integer, SubFigure>();
            for (Element element : annotation.select(ZONE)) {
                SubFigure subFigure = new SubFigure();
                setPoints(subFigure, element);

                subFigures.put(i, subFigure);
                i++;
            }

            Figure figure = new Figure(imageName, subFigures);

            figures.put(imageName, figure);
        }

        return figures;

    }

    //Set points on SubFigure based on XML data
    private static void setPoints(SubFigure subFigure, Element subFigureEl) {
        int i = 1;
        for (Element pointEl : subFigureEl.select(POINT)) {

            Point point = new Point(Integer.parseInt(pointEl.attr(X)), Integer.parseInt(pointEl.attr(Y)));

            switch (i) {
                case 1:
                    subFigure.setTopLeft(point);
                case 2:
                    subFigure.setTopRight(point);
                case 3:
                    subFigure.setBottomLeft(point);
                case 4:
                    subFigure.setBottomRight(point);
            }

            i++;
        }
    }
}
