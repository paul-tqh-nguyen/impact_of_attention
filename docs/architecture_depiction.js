{ // Architecture Depictions

    /*******************/
    /* Misc. Utilities */
    /*******************/
    
    const isUndefined = value => value === void(0);

    const shuffle = (array) => array.sort(() => Math.random() - 0.5);

    const zip = rows => rows[0].map((_,c) => rows.map(row => row[c]));
    
    const createNewElement = (childTag, {classes, attributes, innerHTML}={}) => {
        const newElement = childTag === 'svg' ? document.createElementNS('http://www.w3.org/2000/svg', childTag) : document.createElement(childTag);
        if (!isUndefined(classes)) {
            classes.forEach(childClass => newElement.classList.add(childClass));
        }
        if (!isUndefined(attributes)) {
            Object.entries(attributes).forEach(([attributeName, attributeValue]) => {
                newElement.setAttribute(attributeName, attributeValue);
            });
        }
        if (!isUndefined(innerHTML)) {
            newElement.innerHTML = innerHTML;
        }
        return newElement;
    };

    // D3 Extensions
    d3.selection.prototype.moveToFront = function() {
	return this.each(function() {
	    if (this.parentNode !== null) {
		this.parentNode.appendChild(this);
	    }
	});
    };

    d3.selection.prototype.moveToBack = function() {
        return this.each(function() {
            var firstChild = this.parentNode.firstChild;
            if (firstChild) {
                this.parentNode.insertBefore(this, firstChild);
            }
        });
    };

    /***************************/
    /* Visualization Utilities */
    /***************************/
    
    const innerMargin = 150;
    const textMargin = 8;
    const curvedArrowOffset = 30;

    const xCenterPositionForIndex = (encompassingSvg, index, total, overridingInnerMargin=innerMargin) => {
        const svgWidth = parseFloat(encompassingSvg.style('width'));
        const innerWidth = svgWidth - 2 * overridingInnerMargin;
        const delta = innerWidth / (total - 1);
        const centerX = overridingInnerMargin + index * delta;
        return centerX;
    };

    const generateTextWithBoundingBox = (encompassingSvg, parentGroupClass, textElementClass, boundingBoxClass, textCenterX, yPosition, textString) => {
        const parentGroup = encompassingSvg
              .append('g')
              .classed(parentGroupClass, true);
        const textElement = parentGroup
              .append('text')
	      .attr('y', yPosition)
              .classed(textElementClass, true)
              .html(textString);
        textElement
	    .attr('x', textCenterX - textElement.node().getBBox().width / 2);
        const boundingBoxElement = parentGroup
              .append('rect')
              .classed(boundingBoxClass, true)
              .attr('x', textElement.attr('x') - textMargin)
              .attr('y', () => {
                  const textElementBBox = textElement.node().getBBox();
                  return textElementBBox.y - textMargin;
              })
              .attr('width', textElement.node().getBBox().width + 2 * textMargin)
              .attr('height', textElement.node().getBBox().height + 2 * textMargin);
        textElement.moveToFront();
        return parentGroup;
    };

    const getD3HandleTopXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width/2;
        const y = boundingBox.y;
        return [x, y];
    };

    const getD3HandleBottomXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width/2;
        const y = boundingBox.y + boundingBox.height;
        return [x, y];
    };

    const getD3HandleTopLeftXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x;
        const y = boundingBox.y;
        return [x, y];
    };

    const getD3HandleBottomLeftXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x;
        const y = boundingBox.y + boundingBox.height;
        return [x, y];
    };

    const getD3HandleTopRightXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width;
        const y = boundingBox.y;
        return [x, y];
    };

    const getD3HandleBottomRightXY = (element) => {
        /* element is a D3 handle */
        const boundingBox = element.node().getBBox();
        const x = boundingBox.x + boundingBox.width;
        const y = boundingBox.y + boundingBox.height;
        return [x, y];
    };

    const defineArrowHead = (encompassingSvg) => {
	const defs = encompassingSvg.append('defs');
	const marker = defs.append('marker')
	      .attr('markerWidth', '10')
	      .attr('markerHeight', '10')
	      .attr('refX', '5')
	      .attr('refY', '3')
	      .attr('orient', 'auto')
	      .attr('id', 'arrowhead');
        const polygon = marker.append('polygon')
	      .attr('points', '0 0, 6 3, 0 6');
    };
    
    
    const drawArrow = (encompassingSvg, [x1, y1], [x2, y2]) => {
        const line = encompassingSvg
              .append('line')
	      .attr('marker-end','url(#arrowhead)')
              .moveToBack()
	      .attr('x1', x1)
	      .attr('y1', y1)
	      .attr('x2', x2)
	      .attr('y2', y2)
              .classed('arrow-line', true);
    };
    
    const drawCurvedArrow = (encompassingSvg, [x1, y1], [x2, y2]) => {
	const midpointX = (x1+x2)/2;
	const midpointY = (y1+y2)/2;
	const dx = (x2 - x1);
	const dy = (y2 - y1);
	const normalization = Math.sqrt((dx * dx) + (dy * dy));
	const offSetX = midpointX + curvedArrowOffset*(dy/normalization);
	const offSetY = midpointY - curvedArrowOffset*(dx/normalization);
	const path = `M ${x1}, ${y1} S ${offSetX}, ${offSetY} ${x2}, ${y2}`;
        const line = encompassingSvg
              .append('path')
	      .attr('marker-end','url(#arrowhead)')
              .moveToBack()
	      .attr('d', path)
              .classed('arrow-line', true);
    };
    
    /******************/
    /* Visualizations */
    /******************/
    
    const renderAttentionArchitecture = () => {

        /* Init */
        
        const svg = d3.select('#attention-depiction');
        svg.selectAll('*').remove();
        svg
	    .attr('width', `80vw`)
	    .attr('height', `${1000}px`);
        defineArrowHead(svg);
        const svgWidth = parseFloat(svg.style('width'));

        /* Blocks */

        const words = ['"Films"', '"similar"', '"to"', '&hellip;', '"production."'];
        
        // Words
        const wordGroups = words.map((word, i) => {
            const textCenterX = xCenterPositionForIndex(svg, i, words.length);
            const wordGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', textCenterX, 100, word);
            wordGroup.classed('word-group', true);
            return wordGroup;
        });

        // Embedding Layer
        const embeddingGroups = wordGroups.map(wordGroup => {
            const centerX = getD3HandleTopXY(wordGroup)[0];
            const embeddingGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 200, 'Embedding Layer');
            embeddingGroup.classed('embedding-group', true);
            return embeddingGroup;
        });

        // LSTM Layer
        const LSTMGroups = wordGroups.map(wordGroup => {
            const centerX = getD3HandleTopXY(wordGroup)[0];
            const LSTMGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 300, 'BiLSTM');
            LSTMGroup.classed('lstm-group', true);
            return LSTMGroup;
        });

        // Attention Layer
        const attentionGroups = wordGroups.map(wordGroup => {
            const centerX = getD3HandleTopXY(wordGroup)[0];
            const attentionGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 400, 'Attention');
            attentionGroup.classed('attention-group', true);
            return attentionGroup;
        });

        // Attention Softmax Layer
        const attentionSoftmaxCenterX = svgWidth/2;
        const attentionSoftmaxGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', attentionSoftmaxCenterX, 500, 'Softmax');
        attentionSoftmaxGroup.classed('attention-softmax-group', true);
        const attentionSoftmaxGroupLeftX = attentionGroups[0].node().getBBox().x;
        const rightmostAttentionGroupBoundingBox = attentionGroups[words.length-1].node().getBBox();
        const attentionSoftmaxGroupRightX = rightmostAttentionGroupBoundingBox.x + rightmostAttentionGroupBoundingBox.width;
        attentionSoftmaxGroup.select('rect')
            .attr('x', attentionSoftmaxGroupLeftX)
            .attr('width', attentionSoftmaxGroupRightX - attentionSoftmaxGroupLeftX);

        // Attention Sum Layer
        const attentionSumCenterX = svgWidth/2;
        const attentionSumGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', attentionSumCenterX, 600, '+');
        attentionSumGroup.classed('attention-sum-group', true);
        
        // Fully Connected Layer
        const fullyConnectedCenterX = svgWidth/2;
        const fullyConnectedGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', fullyConnectedCenterX, 700, 'Fully Connected Layer');
        fullyConnectedGroup.classed('fully-connected-group', true);
        
        // Softmax Layer
        const softmaxCenterX = svgWidth/2;
        const softmaxGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', softmaxCenterX, 800, 'Softmax Layer');
        softmaxGroup.classed('softmax-group', true);
        
        // Output Layer
        const negativeScore = Math.random();
        const positiveScore = 1 - negativeScore;
        const outputGroups = [['Negative Score', negativeScore], ['Positive Score', positiveScore]].map(([scoreText, score], i) => {
            const centerX = xCenterPositionForIndex(svg, i, 2, 300);
            const outputGroupLabelText = `${scoreText}: ${score.toFixed(4)}`;
            const outputGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 900, outputGroupLabelText);
            outputGroup.classed('output-group', true);
            return outputGroup;
        });

        /* Arrows */

        // Words to Embedding Layer
        zip([wordGroups, embeddingGroups]).forEach(([wordGroup, embeddingGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(wordGroup), getD3HandleTopXY(embeddingGroup));
        });
        
        // Embedding Layer to LSTM Layer
        zip([embeddingGroups, LSTMGroups]).forEach(([embeddingGroup, LSTMGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(embeddingGroup), getD3HandleTopXY(LSTMGroup));
        });

        // Intra-LSTM Layer Arrows
        LSTMGroups.forEach((LSTMGroup, i) => {
            if (i<LSTMGroups.length-1) {
                const nextLSTMGroup = LSTMGroups[i+1];
                drawCurvedArrow(svg, getD3HandleTopRightXY(LSTMGroup), getD3HandleTopLeftXY(nextLSTMGroup));
                drawCurvedArrow(svg, getD3HandleBottomLeftXY(nextLSTMGroup), getD3HandleBottomRightXY(LSTMGroup));
            }
	});
        
        // LSTM Layer to Attention Layer
        zip([LSTMGroups, attentionGroups]).forEach(([LSTMGroup, attentionGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(LSTMGroup), getD3HandleTopXY(attentionGroup));
        });
        
        // Attention Layer to Softmax Layer
        attentionGroups.forEach((attentionGroup) => {
            const [attentionGroupBottomX, attentionGroupBottomY] = getD3HandleBottomXY(attentionGroup);
            const attentionSoftmaxGroupTopY = getD3HandleTopXY(attentionSoftmaxGroup)[1];
            drawArrow(svg, [attentionGroupBottomX, attentionGroupBottomY], [attentionGroupBottomX, attentionSoftmaxGroupTopY]);
        });
        
        // Softmax Layer to Attention Sum Layer
        attentionGroups.forEach((attentionGroup) => {
            const attentionGroupBottomX = getD3HandleBottomXY(attentionGroup)[0];
            const attentionSoftmaxGroupBottomY = getD3HandleBottomXY(attentionSoftmaxGroup)[1];
            drawArrow(svg, [attentionGroupBottomX, attentionSoftmaxGroupBottomY], getD3HandleTopXY(attentionSumGroup));
        });

        // Attention Sum Layer to Fully Connected Layer
        drawArrow(svg, getD3HandleBottomXY(attentionSumGroup), getD3HandleTopXY(fullyConnectedGroup));

        // Fully Connected Layer to Softmax Layer
        drawArrow(svg, getD3HandleBottomXY(fullyConnectedGroup), getD3HandleTopXY(softmaxGroup));

        // Softmax Layer to Output Layer
        outputGroups.forEach(outputGroup => {
            drawArrow(svg, getD3HandleBottomXY(softmaxGroup), getD3HandleTopXY(outputGroup));
        });
        
    };
    renderAttentionArchitecture();
    window.addEventListener('resize', renderAttentionArchitecture);
    
    const renderVanillaArchitecture = () => {

        /* Init */
        
        const svg = d3.select('#vanilla-depiction');
        svg.selectAll('*').remove();
        svg
	    .attr('width', `80vw`)
	    .attr('height', `${900}px`);
        defineArrowHead(svg);
        const svgWidth = parseFloat(svg.style('width'));

        /* Blocks */

        const words = ['"This"', '"movie"', '"initially"', '&hellip;', '"clarity."'];
        
        // Words
        const wordGroups = words.map((word, i) => {
            const textCenterX = xCenterPositionForIndex(svg, i, words.length);
            const wordGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', textCenterX, 100, word);
            wordGroup.classed('word-group', true);
            return wordGroup;
        });

        // Embedding Layer
        const embeddingGroups = wordGroups.map(wordGroup => {
            const centerX = getD3HandleTopXY(wordGroup)[0];
            const embeddingGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 200, 'Embedding Layer');
            embeddingGroup.classed('embedding-group', true);
            return embeddingGroup;
        });

        // LSTM Layer
        const LSTMGroups = wordGroups.map(wordGroup => {
            const centerX = getD3HandleTopXY(wordGroup)[0];
            const LSTMGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 300, 'BiLSTM');
            LSTMGroup.classed('lstm-group', true);
            return LSTMGroup;
        });

        // Mean Layer
        const meanCenterX = svgWidth/2;
        const meanGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', meanCenterX, 500, 'Mean');
        meanGroup.classed('mean-group', true);
        
        // Fully Connected Layer
        const fullyConnectedCenterX = svgWidth/2;
        const fullyConnectedGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', fullyConnectedCenterX, 600, 'Fully Connected Layer');
        fullyConnectedGroup.classed('fully-connected-group', true);
        
        // Softmax Layer
        const softmaxCenterX = svgWidth/2;
        const softmaxGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', softmaxCenterX, 700, 'Softmax Layer');
        softmaxGroup.classed('softmax-group', true);
        
        // Output Layer
        const negativeScore = Math.random();
        const positiveScore = 1 - negativeScore;
        const outputGroups = [['Negative Score', negativeScore], ['Positive Score', positiveScore]].map(([scoreText, score], i) => {
            const centerX = xCenterPositionForIndex(svg, i, 2, 300);
            const outputGroupLabelText = `${scoreText}: ${score.toFixed(4)}`;
            const outputGroup = generateTextWithBoundingBox(svg, 'text-with-bbox-group', 'text-with-bbox-group-text', 'text-with-bbox-group-bounding-box', centerX, 800, outputGroupLabelText);
            outputGroup.classed('output-group', true);
            return outputGroup;
        });

        /* Arrows */

        // Words to Embedding Layer
        zip([wordGroups, embeddingGroups]).forEach(([wordGroup, embeddingGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(wordGroup), getD3HandleTopXY(embeddingGroup));
        });
        
        // Embedding Layer to LSTM Layer
        zip([embeddingGroups, LSTMGroups]).forEach(([embeddingGroup, LSTMGroup]) => {
            drawArrow(svg, getD3HandleBottomXY(embeddingGroup), getD3HandleTopXY(LSTMGroup));
        });

        // Intra-LSTM Layer Arrows
        LSTMGroups.forEach((LSTMGroup, i) => {
            if (i<LSTMGroups.length-1) {
                const nextLSTMGroup = LSTMGroups[i+1];
                drawCurvedArrow(svg, getD3HandleTopRightXY(LSTMGroup), getD3HandleTopLeftXY(nextLSTMGroup));
                drawCurvedArrow(svg, getD3HandleBottomLeftXY(nextLSTMGroup), getD3HandleBottomRightXY(LSTMGroup));
            }
	});
        
        // LSTM Layer to Mean Layer
        LSTMGroups.forEach(LSTMGroup => {
            drawArrow(svg, getD3HandleBottomXY(LSTMGroup), getD3HandleTopXY(meanGroup));
        });
        
        // Mean Layer to Fully Connected Layer
        drawArrow(svg, getD3HandleBottomXY(meanGroup), getD3HandleTopXY(fullyConnectedGroup));

        // Fully Connected Layer to Softmax Layer
        drawArrow(svg, getD3HandleBottomXY(fullyConnectedGroup), getD3HandleTopXY(softmaxGroup));

        // Softmax Layer to Output Layer
        outputGroups.forEach(outputGroup => {
            drawArrow(svg, getD3HandleBottomXY(softmaxGroup), getD3HandleTopXY(outputGroup));
        });
        
    };
    renderVanillaArchitecture();
    window.addEventListener('resize', renderVanillaArchitecture);

}
